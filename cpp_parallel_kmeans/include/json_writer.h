#pragma once

#include <string>
#include <vector>
#include <map>

// Простой JSON writer для сохранения результатов
class JsonWriter {
public:
    static void write_results(const std::vector<std::map<std::string, std::string>>& results,
                             const std::string& filename);
    
    static std::string escape_json(const std::string& str);
};

