#pragma once

#ifndef AS_STRING

#define AS_STRING(...) AS_STRING2(__VA_ARGS__)
#define AS_STRING2(...) #__VA_ARGS__

#endif

#ifndef REQUIRES

#ifdef __clang__
#define REQUIRES(...) __attribute__((enable_if(__VA_ARGS__, AS_STRING(__VA_ARGS__))))
#elif __GNUC__
#define REQUIRES(...) requires(__VA_ARGS__)
#endif

#define DECL_AND_RETURN(...)                                                                                                                         \
  ->decltype(__VA_ARGS__) { return __VA_ARGS__; }

#endif
