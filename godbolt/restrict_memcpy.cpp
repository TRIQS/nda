
void f(double *__restrict p, double *q, long N) {
  for (long u = 0; u < N; ++u) p[u] = q[u];
}
