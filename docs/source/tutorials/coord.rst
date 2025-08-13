Train a Coordination Policy
===========================

In this tutorial, you will learn how to train a coordination policy to enable effective collaboration between two agents.

Training a coordination policy is very similar to training a single agent, thanks to Duo's standardized abstractions for algorithms, policies, and environments. The main differences are the configuration arguments and the need to wrap the base environments with `CoordEnv`.

We will reuse the script `examples/procgen_yrc.py <https://github.com/khanhptnk/duo-ai/blob/main/examples/procgen_yrc.py>`_.

0. Refresher: What is CoordEnv?
-------------------------------

A `CoordEnv` represents the POMDP presented to the coordination policy.  
It is implemented as a Gym environment and comprises a base environment, a novice, and an expert policy.

The action space contains two actions: ``NOVICE`` and ``EXPERT``, corresponding to querying the novice or the expert for the next decision.  
When an action is chosen, the corresponding agent is queried for a base environment action. This environment action is then fed into the base environment to obtain the next state and reward.

1. Configuration
----------------

Compared to training an agent, training a coordination policy differs in:

- The algorithm, policy, and policy model

- The coordination configuration

- The paths to load the novice and expert agents

Let's look at an example at `configs/procgen_ppo.yaml <https://github.com/khanhptnk/duo-ai/blob/main/configs/procgen_ppo.yaml>`_, which uses the PPO algorithm:

.. code-block:: yaml

    env:
      name: "procgen"
      train:
        distribution_mode: "hard"

    algorithm: 
      name: "ppo"
      total_timesteps: 15000000

    policy:
      name: "ppo"
      model:
        name: "impala_coord_ppo"
        feature_type: obs

    coordination:
      expert_query_cost_weight: 0.4
      switch_agent_cost_weight: 0.0
      temperature: 1.0

    train_novice: "experiments/procgen_novice/best_test.ckpt"
    train_expert: "experiments/procgen_expert/best_test.ckpt"
    test_novice: "experiments/procgen_novice/best_test.ckpt"
    test_expert: "experiments/procgen_expert/best_test.ckpt"


2. Create CoordEnv
------------------

There is a new step in the training script, which creates the `CoordEnv`:

.. code-block:: python

    base_envs = make_base_envs(config)
    # NEW STEP: create CoordEnv
    envs = make_coord_envs(config, base_envs)
    policy = duo_ai.make_policy(config.policy, envs["train"])
    algorithm = duo_ai.make_algorithm(config.algorithm)

    validators = {}
    for split in splits:
        if split != "train":
            validators[split] = duo_ai.Evaluator(config.evaluation, envs[split])

    algorithm.train(policy, envs["train"], validators)

The ``make_coord_envs`` function is implemented as follows:

.. code-block:: python

    def make_coord_envs(config, base_envs):
        # 1) Load novice and expert
        some_base_env = list(base_envs.values())[0]
        train_novice = duo_ai.load_policy(config.train_novice, some_base_env)
        train_expert = duo_ai.load_policy(config.train_expert, some_base_env)
        test_novice = duo_ai.load_policy(config.test_novice, some_base_env)
        test_expert = duo_ai.load_policy(config.test_expert, some_base_env)

        # 2) Create CoordEnv
        # We use train_novice and train_expert for training and validation
        # and test_novice and test_expert for testing
        envs = {}
        for split in splits:
            if split in ["train", "val_sim"]:
                novice, expert = train_novice, train_expert
            else:
                novice, expert = test_novice, test_expert
            envs[split] = duo_ai.CoordEnv(
                config.coordination, base_envs[split], novice, expert
            )

        # 3) Set coordination costs 
        # compute_reward_per_action() is a user-defined function that computes the cost-per-step 
        # of leverging expert
        # See `Core concepts -> Problem setting` to understand how this cost is intergrated into 
        # the environment reward 
        base_penalty = compute_reward_per_action(config.env)
        for split in splits:
            envs[split].set_costs(base_penalty) 
        return envs


3. Run the script
-----------------

We provide the checkpoints of the novice and expert in the Github repo. You can simply run this command to train the coordination policy:

.. code-block:: bash

    python examples/procgen_yrc.py \
        --config configs/procgen_ppo.yaml \
        --mode train \
        --type coord \
        overwrite=1


Here is the expected result:

.. code-block:: bash

    [3:29:56 INFO]: BEST test so far
    [3:29:56 INFO]:    Steps:         16242
      Episode length: mean   63.45  min   18.00  max  208.00
      Reward:         mean 7.07 ± 0.52
      Base Reward:    mean 7.58 ± 0.52
      Action 1 fraction:    0.18

`Base reward` refers the raw reward obtained from the base environment, i.e., without the cost of expert assistance. It is always greater than or equal to `Reward`.

You can compare with our `Wandb Log <https://wandb.ai/kxnguyen/YRC-public/runs/7b0imagl?nw=nwuserkxnguyen>`_ to make sure the code runs as expected. 
