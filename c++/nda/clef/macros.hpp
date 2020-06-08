#pragma once

#ifndef AS_STRING

#define AS_STRING(...) AS_STRING2(__VA_ARGS__)
#define AS_STRING2(...) #__VA_ARGS__

#endif

#ifndef CLEF_REQUIRES

#ifdef __clang__
#define CLEF_REQUIRES(...) __attribute__((enable_if(__VA_ARGS__, AS_STRING(__VA_ARGS__))))
#elif __GNUC__
#define CLEF_REQUIRES(...) requires(__VA_ARGS__)
#endif

#define DECL_AND_RETURN(...)                                                                                                                         \
  ->decltype(__VA_ARGS__) { return __VA_ARGS__; }

#endif
