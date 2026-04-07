import gymnasium as gym
names = sorted([str(k) for k in gym.registry.keys() if str(k).startswith('ALE/') and str(k).endswith('-v5')])
print('COUNT', len(names))
for n in names:
    print(n)