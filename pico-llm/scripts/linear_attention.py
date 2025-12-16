import matplotlib.pyplot as plt

softmax_rows = [{'L': 256, 'prefill_ms': 78.13295800000009, 'decode_ms_per_token': 6.036419166666666, 'cache': 'KV-history'}, {'L': 512, 'prefill_ms': 103.32566700000001, 'decode_ms_per_token': 6.6685268066666685, 'cache': 'KV-history'}, {'L': 1024, 'prefill_ms': 214.7890840000004, 'decode_ms_per_token': 8.367484306666668, 'cache': 'KV-history'}, {'L': 2048, 'prefill_ms': 488.9266249999995, 'decode_ms_per_token': 14.293931250000002, 'cache': 'KV-history'}, {'L': 4096, 'prefill_ms': 1604.020417000001, 'decode_ms_per_token': 23.93154541666667, 'cache': 'KV-history'}]
linear_rows = [{'L': 256, 'prefill_ms': 324.584875, 'decode_ms_per_token': 5.610883196666667, 'cache': 'DeltaKet-state(S,Z)'}, {'L': 512, 'prefill_ms': 654.418292, 'decode_ms_per_token': 5.404815833333334, 'cache': 'DeltaKet-state(S,Z)'}, {'L': 1024, 'prefill_ms': 1190.1335839999997, 'decode_ms_per_token': 5.489106526666667, 'cache': 'DeltaKet-state(S,Z)'}, {'L': 2048, 'prefill_ms': 2392.4487499999996, 'decode_ms_per_token': 5.749228609999998, 'cache': 'DeltaKet-state(S,Z)'}, {'L': 4096, 'prefill_ms': 4840.354082999999, 'decode_ms_per_token': 5.5112581933333376, 'cache': 'DeltaKet-state(S,Z)'}]

L = [r["L"] for r in softmax_rows]
softmax = [r["decode_ms_per_token"] for r in softmax_rows]
linear = [r["decode_ms_per_token"] for r in linear_rows]

plt.figure(figsize=(10,6))
plt.plot(L, softmax, marker='o', label='Softmax (KV-history)')
plt.plot(L, linear, marker='o', label='Linear (DeltaKet-state S,Z)')

# Log-2 x axis but show ticks as raw lengths (256, 512, ...)
plt.xscale('log', base=2)
plt.xticks(L, [str(x) for x in L])

plt.xlabel('Context length L')
plt.ylabel('Decode time (ms/token)')
plt.title('Decode scaling: Softmax vs DeltaKet Linear Attention')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
