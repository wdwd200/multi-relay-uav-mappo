# Stage 4.3 P3/P4 implementation audit

This is an engineering-contract audit only; it is not performance evidence.

- P3 (`topology_info`): 26 local-observation values plus fixed chain topology = 97 inputs for K=4.
- P4 (`graph`): node=6, edge=7, hidden=32, shared chain message passing; Actor input = 58 for K=4.
- P3 parameters: 29446; P4 parameters: 29094; relative gap: 0.011954.
- Environment, Reward, 26-D observation, 47-D Critic, PPO/GAE, and `stage4-formal` are unchanged.
- V0H0 remains fixed: value normalization disabled, gamma=0.99, lambda=0.95.
