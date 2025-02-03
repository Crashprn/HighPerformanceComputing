#pragma once

#include <string>
#include <vector>
#include <cstdint>


class Student
{
    public:
        std::string m_name;
        
        Student(std::string name, std::vector<std::string> classes, std::vector<std::uint8_t> grades);

        std::vector<std::string> getClasses() const;
        std::vector<std::uint8_t> getGrades() const;
        std::string getName() const;

    private:
        void setClasses(std::vector<std::string> classes);
        void setGrades(std::vector<std::uint8_t> grades);
        void setName(std::string name);

        std::vector<std::string> m_classes;
        std::vector<std::uint8_t> m_grades;


};