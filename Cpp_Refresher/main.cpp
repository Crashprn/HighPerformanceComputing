#include <iostream>
#include <vector>
#include <string>

#include "student.hpp"

struct Student_Alternate {
    std::string name;
    std::vector<std::string> classes;
    std::vector<std::uint8_t> grades;
};

int main() {
    Student student1("John Doe", {"Math", "Science", "History"}, {85, 90, 78});
    Student student2("Jane Smith", {"Math", "Science", "History"}, {92, 88, 95});
    Student student3("Alice Johnson", {"Math", "Science", "History"}, {70, 75, 80});

    std::vector<Student> students = {student1, student2, student3};

    for (decltype(students.size()) i = 0; i < students.size(); ++i) {
        std::cout << "Student " << i + 1 << ": " << students[i].m_name << std::endl;
        std::cout << "Classes: ";
        for (const auto& className : students[i].getClasses()) {
            std::cout << className << " ";
        }
        std::cout << std::endl;
        std::cout << "Grades: ";
        for (const auto& grade : students[i].getGrades()) {
            std::cout << static_cast<int>(grade) << " ";
        }
        std::cout << std::endl;

        int a = 3;

        int* p = &a;

        std::cout << "Address of a: " << p << std::endl;
        std::cout << "Value of a: " << *p << std::endl;
        std::cout << "Address of p: " << &p << std::endl;

    }

    int* b = new int;

    *b = 5;

    std::cout << "Value of b: " << *b << std::endl;

    delete b;

    Student* student4 = &student1;

    std::cout << "Student 4: " << student4->m_name << std::endl; 

    Student_Alternate student5;

    int anarray[10];

    for (int i = 0; i < 10; ++i) {
        anarray[i] = i;
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << anarray[i] << std::endl;
    }

    int anarray2[10][10];

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            anarray2[i][j] = i * j;
        }
    }

    


    return 0;
}