#include <gtest/gtest.h> // NOLINT
#include <memory>

#define NDA_DEBUG_MEMORY

#include <nda/storage/handle.hpp>

using namespace nda::mem;

//---------------------------------------------

// make sure that all memory is deallocated at the end of each test
class Ref : public ::testing::Test {
  protected:
  void TearDown() override {
  }
};

TEST_F(Ref, HR) { // NOLINT

  handle_heap<int, void> h{10};

  auto h2 = handle_heap<int, void>{10};

  // make sure it is a copy
  h.data()[2] = 89;
  handle_heap<int, void> h3{h};
  h.data()[2] = 0;
  EXPECT_EQ(h3.data()[2], 89); //NOLINT
}

// ---- Construct R, S
TEST_F(Ref, HSR) { // NOLINT

  handle_heap<int, void> h{10};

  handle_shared<int> s{h};

  EXPECT_EQ(s.refcount(), 2); //NOLINT
}

// --- implicit construction

// ---- More complex
TEST_F(Ref, HSRS) { // NOLINT

  handle_heap<int, void> h{10};

  handle_shared<int> s{h};
  EXPECT_EQ(s.refcount(), 2); //NOLINT

  s = handle_shared<int>{h};
  EXPECT_EQ(s.refcount(), 2); //NOLINT

  handle_shared<int> s2{h};
  s = s2;
  EXPECT_EQ(s.refcount(), 3); //NOLINT
}

// ---- check with something that is constructed/destructed.
struct Number {
  int u               = 9;
  static inline int c = 0;
  Number() {
    c++;
    std::cerr << "Constructing Number \n";
  };
  ~Number() {
    c--;
    std::cerr << "Destructing Number \n";
  };
};

TEST_F(Ref, HR_with_cd) { // NOLINT
  { handle_heap<Number, void> h{5}; }
  EXPECT_EQ(Number::c, 0); //NOLINT
}

// --- check with a shared_ptr

// TO BE REWRITTEN ? NEEDED ?
void release_sp(void *x) {
  auto *p = (std::shared_ptr<Number> *)x; // NOLINT
  p->reset();
  delete p; // NOLINT
}

TEST_F(Ref, HR_with_sharedPtr) { // NOLINT
  {
    handle_shared<Number> s;
  }
  EXPECT_EQ(Number::c, 0); //NOLINT
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
