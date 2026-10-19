// tokenizer.h — pure-C++ gemma-4 BPE tokenizer (loads HF tokenizer.json). No runtime deps beyond json.hpp.
// Spec: normalize ' '->U+2581(▁); split to unicode chars; byte_fallback (char not in vocab -> <0xXX>=238+byte);
// merge adjacent pair with lowest rank until none; decode reverses (▁->space, <0xXX>->raw byte, fuse).
#pragma once
#include "third_party/json.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstdint>
#include <climits>

struct Tokenizer {
    std::unordered_map<std::string,int> vocab;              // token -> id
    std::vector<std::string> id2tok;                        // id -> token
    std::unordered_map<uint64_t, std::pair<int,int>> merges;// (idA,idB) -> (rank, mergedId)
    std::vector<std::pair<std::string,int>> specials;       // (content,id), matched literally before BPE
    int eos_id=1, bos_id=2, turn_start=105, turn_end=106;   // gemma-4: <|turn>=105, <turn|>=106
    int chan_start=100, chan_end=101;                        // <|channel>=100, <channel|>=101
    int tool_start=46, tool_end=47, tcall_start=48, tcall_end=49;   // <|tool>/<tool|> (decl), <|tool_call>/<tool_call|>
    int think_id=98, tresp_start=50, tresp_end=51;                  // <|think|> (triggers CoT), <|tool_response>/<tool_response|>
    std::vector<int> stop_ids;                               // generation stops: <eos>, <turn|>, <|tool_response>=50

    static uint64_t pk(int a,int b){ return ((uint64_t)(uint32_t)a<<32)|(uint32_t)(uint32_t)b; }
    static int u8len(unsigned char c){ return c<0x80?1: (c>>5)==0x6?2: (c>>4)==0xE?3: (c>>3)==0x1E?4:1; }

    void load(const std::string& path){
        std::ifstream f(path); nlohmann::json j; f>>j;
        auto& v=j["model"]["vocab"];
        id2tok.assign(v.size(), std::string());
        for(auto it=v.begin(); it!=v.end(); ++it){ int id=it.value().get<int>(); const std::string& s=it.key();
            vocab[s]=id; if(id>=0 && id<(int)id2tok.size()) id2tok[id]=s; }
        int rank=0;
        for(auto& m : j["model"]["merges"]){
            std::string a,b;
            if(m.is_array()){ a=m[0].get<std::string>(); b=m[1].get<std::string>(); }
            else { std::string s=m.get<std::string>(); size_t sp=s.find(' '); a=s.substr(0,sp); b=s.substr(sp+1); }
            auto ia=vocab.find(a), ib=vocab.find(b); auto ic=vocab.find(a+b);
            if(ia!=vocab.end()&&ib!=vocab.end()&&ic!=vocab.end()) merges[pk(ia->second,ib->second)]={rank, ic->second};
            rank++;
        }
        if(j.contains("added_tokens")) for(auto& at : j["added_tokens"]){ std::string c=at["content"].get<std::string>(); int id=at["id"].get<int>();
            specials.push_back({c,id}); if(c=="<eos>")eos_id=id; else if(c=="<bos>")bos_id=id; }
        auto ts=vocab.find("<|turn>"), te=vocab.find("<turn|>");   // gemma-4 turn markers (in main vocab, not added_tokens)
        if(ts!=vocab.end()) turn_start=ts->second;
        if(te!=vocab.end()) turn_end=te->second;
        auto cs=vocab.find("<|channel>"), ce=vocab.find("<channel|>");
        if(cs!=vocab.end()) chan_start=cs->second;
        if(ce!=vocab.end()) chan_end=ce->second;
        auto t0=vocab.find("<|tool>"), t1=vocab.find("<tool|>"), t2=vocab.find("<|tool_call>"), t3=vocab.find("<tool_call|>");
        if(t0!=vocab.end())tool_start=t0->second; if(t1!=vocab.end())tool_end=t1->second;
        if(t2!=vocab.end())tcall_start=t2->second; if(t3!=vocab.end())tcall_end=t3->second;
        auto tk98=vocab.find("<|think|>"), tr0=vocab.find("<|tool_response>"), tr1=vocab.find("<tool_response|>");
        if(tk98!=vocab.end())think_id=tk98->second; if(tr0!=vocab.end())tresp_start=tr0->second; if(tr1!=vocab.end())tresp_end=tr1->second;
        stop_ids={eos_id, turn_end}; auto tr=vocab.find("<|tool_response>"); if(tr!=vocab.end()) stop_ids.push_back(tr->second);
    }
    // gemma-4 chat prompt. msgs=[(role,content)]. enable_thinking: false pre-fills an empty thought channel
    // (straight to answer, fast); true lets the model reason in <|channel>thought..<channel|> (parsed to reasoning_content).
    std::vector<int> chat_prompt(const std::vector<std::pair<std::string,std::string>>& msgs, bool enable_thinking=false,
                                 const std::vector<std::string>& tool_decls={}){
        std::vector<int> ids={bos_id};
        bool has_sys = !msgs.empty() && msgs[0].first=="system";
        if(has_sys || !tool_decls.empty() || enable_thinking){   // first system turn: <|think|> marker + system content + tool decls
            ids.push_back(turn_start); encode_text("system\n",ids);
            if(enable_thinking){ ids.push_back(think_id); encode_text("\n",ids); }   // <|think|> triggers real CoT (gemma-4 fixed template)
            if(has_sys) encode_text(msgs[0].second,ids);
            for(auto& d : tool_decls){ ids.push_back(tool_start); encode_text(d,ids); ids.push_back(tool_end); }
            ids.push_back(turn_end); encode_text("\n",ids);
        }
        for(size_t i=(has_sys?1:0); i<msgs.size(); ++i){ auto& m=msgs[i]; std::string role=(m.first=="assistant")?"model":m.first;
            ids.push_back(turn_start); encode_text(role+"\n",ids); encode_text(m.second,ids); ids.push_back(turn_end); encode_text("\n",ids); }
        ids.push_back(turn_start); encode_text("model\n",ids);   // generation prompt — no channel stub (patched gemma-4 behavior)
        return ids;
    }
    bool is_stop(int id){ for(int s:stop_ids) if(id==s) return true; return false; }
    // BPE a normalized (spaces already ▁) plain segment
    void bpe(const std::string& t, std::vector<int>& out){
        std::vector<int> s; s.reserve(t.size());
        for(size_t i=0;i<t.size();){ int L=u8len((unsigned char)t[i]); std::string ch=t.substr(i,L);
            auto it=vocab.find(ch);
            if(it!=vocab.end()) s.push_back(it->second);
            else for(int b=0;b<L;++b) s.push_back(238+(unsigned char)t[i+b]);   // byte_fallback <0xXX>
            i+=L; }
        while(s.size()>=2){
            int bestK=-1, bestRank=INT_MAX, bestMerged=-1;
            for(size_t k=0;k+1<s.size();++k){ auto m=merges.find(pk(s[k],s[k+1]));
                if(m!=merges.end() && m->second.first<bestRank){ bestRank=m->second.first; bestK=(int)k; bestMerged=m->second.second; } }
            if(bestK<0) break;
            s[bestK]=bestMerged; s.erase(s.begin()+bestK+1);
        }
        for(int id : s) out.push_back(id);
    }
    // normalize + BPE a non-special text segment
    void encode_text(const std::string& text, std::vector<int>& out){
        std::string n; n.reserve(text.size()+8);
        for(char c : text){ if(c==' ') n+="\xE2\x96\x81"; else n.push_back(c); }   // ' ' -> ▁(U+2581)
        bpe(n, out);
    }
    // full encode: extract specials literally, BPE the rest
    std::vector<int> encode(const std::string& text, bool add_bos=false){
        std::vector<int> out; if(add_bos) out.push_back(bos_id);
        size_t i=0;
        while(i<text.size()){
            size_t bestPos=std::string::npos; int bestId=-1; size_t bestLen=0;
            for(auto& sp : specials){ size_t p=text.find(sp.first, i);
                if(p!=std::string::npos && (p<bestPos || (p==bestPos && sp.first.size()>bestLen))){ bestPos=p; bestId=sp.second; bestLen=sp.first.size(); } }
            if(bestPos==std::string::npos){ encode_text(text.substr(i), out); break; }
            if(bestPos>i) encode_text(text.substr(i,bestPos-i), out);
            out.push_back(bestId); i=bestPos+bestLen;
        }
        return out;
    }
    std::string decode(const std::vector<int>& ids, bool skip_special=true){
        std::string out;
        for(int id : ids){
            if(id<0||id>=(int)id2tok.size()) continue;
            const std::string& s=id2tok[id];
            if(s.size()==6 && s[0]=='<'&&s[1]=='0'&&s[2]=='x'&&s[5]=='>'){   // <0xXX> -> raw byte (fuse)
                auto hx=[](char c){ return c<='9'?c-'0':(c|32)-'a'+10; };
                out.push_back((char)((hx(s[3])<<4)|hx(s[4]))); continue; }
            if(skip_special && s.size()>=3 && s[0]=='<' && (s[1]=='|' || s[s.size()-2]=='|' || s=="<bos>"||s=="<eos>"||s=="<pad>")) continue;  // gemma-4 control tokens <|X> / <X|>
            for(size_t k=0;k<s.size();){ if(k+2<s.size()&&(unsigned char)s[k]==0xE2&&(unsigned char)s[k+1]==0x96&&(unsigned char)s[k+2]==0x81){ out.push_back(' '); k+=3; }
                else { out.push_back(s[k]); ++k; } }
        }
        return out;
    }
};
