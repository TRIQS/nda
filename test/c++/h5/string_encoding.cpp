#include <gtest/gtest.h>

#include <h5/h5.hpp>

using namespace h5;

TEST(H5, Encoding) {

  std::string ascii_str = "Hello World!";
  std::string utf8_str  = "Price: 10 €";

  {
    h5::file file("encoding.h5", 'w');

    // Store ASCII
    h5_write(file, "ASCII", ascii_str);
    h5_write_attribute(file, "ASCII_Attr", ascii_str);

    // Store UTF8
    h5_write(file, "UTF8", utf8_str);
    h5_write_attribute(file, "UTF8_Attr", utf8_str);
  }

  {
    h5::file file("encoding.h5", 'r');

    // Read ASCII
    std::string ascii_str_read = "";
    h5_read(file, "ASCII", ascii_str_read);
    EXPECT_EQ(ascii_str, ascii_str_read);
    h5_read_attribute(file, "ASCII_Attr", ascii_str_read);
    EXPECT_EQ(ascii_str, ascii_str_read);

    // Read UTF8
    std::string utf8_str_read = "";
    h5_read(file, "UTF8", utf8_str_read);
    EXPECT_EQ(utf8_str, utf8_str_read);
    h5_read_attribute(file, "UTF8_Attr", utf8_str_read);
    EXPECT_EQ(utf8_str, utf8_str_read);
  }
}
