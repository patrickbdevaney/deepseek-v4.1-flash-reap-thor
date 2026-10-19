// npy.h — minimal .npy reader for the V4.1 unit gates (C-order, little-endian, numeric dtypes).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Npy {
    std::string descr;            // e.g. "<f4", "|u1", "<i4", "<u2"
    std::vector<size_t> shape;
    std::vector<uint8_t> data;
    size_t count() const { size_t n = 1; for (size_t d : shape) n *= d; return n; }
    template <class T> const T* as() const { return reinterpret_cast<const T*>(data.data()); }
};

inline Npy npy_load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[6]; f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("not npy: " + path);
    uint8_t ver[2]; f.read((char*)ver, 2);
    uint32_t hl = 0;
    if (ver[0] == 1) { uint16_t h; f.read((char*)&h, 2); hl = h; } else f.read((char*)&hl, 4);
    std::string h(hl, '\0'); f.read(&h[0], hl);
    Npy n;
    size_t a = h.find("'descr': '") + 10; n.descr = h.substr(a, h.find('\'', a) - a);
    if (h.find("'fortran_order': True") != std::string::npos) throw std::runtime_error("fortran order: " + path);
    size_t s = h.find('(', h.find("'shape'")) + 1, e = h.find(')', s);
    std::string sh = h.substr(s, e - s);
    for (size_t p = 0; p < sh.size();) {
        while (p < sh.size() && (sh[p] == ' ' || sh[p] == ',')) ++p;
        if (p >= sh.size()) break;
        n.shape.push_back(strtoull(sh.c_str() + p, nullptr, 10));
        while (p < sh.size() && sh[p] != ',') ++p;
    }
    const size_t isz = (size_t)atoi(n.descr.c_str() + 2);
    n.data.resize(n.count() * isz);
    f.read((char*)n.data.data(), n.data.size());
    if (!f) throw std::runtime_error("short read: " + path);
    return n;
}
