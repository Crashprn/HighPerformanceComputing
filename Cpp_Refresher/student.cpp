#include "student.hpp"

Student::Student(std::string name, std::vector<std::string> classes, std::vector<std::uint8_t> grades)
{
    setName(name);
    setClasses(classes);
    setGrades(grades);
}

std::vector<std::string> Student::getClasses() const
{
    return m_classes;
}

std::vector<std::uint8_t> Student::getGrades() const
{
    return m_grades;
}
std::string Student::getName() const
{
    return m_name;
}



void Student::setClasses(std::vector<std::string> classes)
{
    m_classes = classes;
}

void Student::setGrades(std::vector<std::uint8_t> grades)
{
    m_grades = grades;
}

void Student::setName(std::string name)
{
    m_name = name;
}

