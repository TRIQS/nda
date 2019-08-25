#include <gtest/gtest.h> // NOLINT
#include <memory>
 
#define NDA_DEBUG_MEMORY

#include <nda/storage/handle.hpp>
 
using namespace nda::mem;

void print(rtable_t const &) {}

// test the rtable
TEST(rtable, base) { // NOLINT

  rtable_t ta(5);

  print(ta);

  auto c1 = ta.get();
  EXPECT_EQ(c1, 1); //NOLINT
  print(ta);

  auto c2 = ta.get();
  EXPECT_EQ(c2, 2); //NOLINT
  print(ta);

  ta.incref(c1);
  EXPECT_EQ(ta.refcounts()[c1], 2); //NOLINT
  print(ta);

  ta.decref(c1);
  print(ta);

  auto c3 = ta.get();
  EXPECT_EQ(c3, 3); //NOLINT
  print(ta);

  ta.decref(c1);
  print(ta);

  auto c4 = ta.get();
  EXPECT_EQ(c4, 1); //NOLINT
  print(ta);

  auto c5 = ta.get();
  EXPECT_EQ(c5, 4); //NOLINT
  print(ta);

  auto c6 = ta.get();
  EXPECT_EQ(c6, 5); //NOLINT
  print(ta);

  ta.decref(c6);
  print(ta);
  ta.decref(c5);
  print(ta);
  ta.decref(c4);
  print(ta);
  ta.decref(c3);
  print(ta);
  ta.decref(c2);
  print(ta);
}

//---------------------------------------------

// make sure that all memory is deallocated at the end of each test
class Ref : public ::testing::Test {
  protected:
  void TearDown() override {
    //EXPECT_TRUE(globals::rtable.empty());//NOLINT
    // EXPECT_TRUE(globals::alloc.empty());//NOLINT
  }
};

TEST_F(Ref, HR) { // NOLINT

  handle<int, 'R'> h{10};

  auto h2 = handle<int, 'R'>{10};

  // make sure it is a copy
  h.data()[2] = 89;
  handle<int, 'R'> h3{h};
  h.data()[2] = 0;
  EXPECT_EQ(h3.data()[2], 89); //NOLINT
}

// ---- Contruct R B
TEST_F(Ref, HBR) { // NOLINT

  handle<int, 'R'> h{10};

  handle<int, 'B'> b{h};
  handle<int, 'B'> b2;
  b2 = h;

  // make sure it is a copy
  b.data()[2] = 89;
  handle<int, 'R'> h2{b};
  b.data()[2] = 0;
  EXPECT_EQ(h2.data()[2], 89); //NOLINT
}

// ---- Construct R, S
TEST_F(Ref, HSR) { // NOLINT

  handle<int, 'R'> h{10};

  handle<int, 'S'> s{h};

  EXPECT_EQ(s.refcount(), 2); //NOLINT
}

// --- implicit construction

// ---- More complex
TEST_F(Ref, HSRS) { // NOLINT

  handle<int, 'R'> h{10};

  handle<int, 'S'> s{h};
  EXPECT_EQ(s.refcount(), 2); //NOLINT

  s = handle<int, 'S'>{h};
  EXPECT_EQ(s.refcount(), 2); //NOLINT

  handle<int, 'S'> s2{h};
  s = s2;
  EXPECT_EQ(s.refcount(), 3); //NOLINT
}

// ---- check with something that is constructed/destructed.
struct Number {
  int u;
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
  { handle<Number, 'R'> h{5}; }
  EXPECT_EQ(Number::c, 0); //NOLINT
}

// --- check with a shared_ptr

// TO BE REWRITTEN ? NEEDED ?
void release_sp(void *x) {
  auto *p = (std::shared_ptr<Number> *)x;
  p->reset();
  delete p;
}

TEST_F(Ref, HR_with_sharedPtr) { // NOLINT
  {
    handle<Number, 'S'> s;
    //s.id = globals::rtable.get();
    //s.sptr = (void *)new std::shared_ptr<Number>{new Number{}};
    //s.release_fnt = (void*)release_sp;
  }
  EXPECT_EQ(Number::c, 0); //NOLINT
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
