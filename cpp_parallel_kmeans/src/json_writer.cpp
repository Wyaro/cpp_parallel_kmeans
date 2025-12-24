#include "../include/json_writer.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <vector>
#include <map>

std::string JsonWriter::escape_json(const std::string& str) {
    std::ostringstream o;
    for (char c : str) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

void JsonWriter::write_results(const std::vector<std::map<std::string, std::string>>& results,
                               const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + filename);
    }
    
    for (const auto& result : results) {
        file << "{";
        bool first = true;
        for (const auto& pair : result) {
            if (!first) file << ",";
            file << "\"" << escape_json(pair.first) << "\":";
            
            // Проверяем, является ли значение числом
            bool is_number = true;
            if (!pair.second.empty()) {
                bool has_dot = false;
                for (char c : pair.second) {
                    if (c == '.' && !has_dot) {
                        has_dot = true;
                    } else if (c < '0' || c > '9') {
                        is_number = false;
                        break;
                    }
                }
            }
            
            if (is_number && !pair.second.empty()) {
                file << pair.second;
            } else {
                file << "\"" << escape_json(pair.second) << "\"";
            }
            first = false;
        }
        file << "}\n";
    }
    
    file.close();
}

