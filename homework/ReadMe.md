# Tryout of DRL algorithms

This directory is my own implementation of popular DRL algorithms inspired by [spinningup](https://spinningup.openai.com).

## Vanilla Policy Gradient



## Trust Region Policy Gradient

## Proximal Policy Gradient

## Deep Deterministic Policy Gradient

Off policy, deterministic policy algorithm

- Replay buffer needs to be large enough to contain all history transactions

- When training the Q network, the target 
$r+{\gamma}(1-d)Q_{\phi_{targ}}(s', \mu_{\theta_{targ}}(s'))$
should be treated as a constant, which means **when compute the gradient of $L_Q$ over $\phi$, the gradient should stop at this term.**

- Target networks and main networks can be separated through using the `tf.variable_scope()`. Another method is using `copy()`(shallow copy in python) to explicitly set up two networks.




## Twin Delayed DDPG
    
Very similar to DDPG but with following quirks:

- Two Q networks (two critics) trained separately [Code lines 49-70]
- Noisy target policy action used to smooth the policy [Code lines 83-113]
- Policy network is updated slower than the Q network to add damps for the sake of stability [Code lines 362]

## Soft Actor Critic




