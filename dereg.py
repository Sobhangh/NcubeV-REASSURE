def unregister_env(env_id: str) -> bool:
    # Gymnasium first
    try:
        from gymnasium.envs.registration import registry as gymnasium_registry
        if env_id in gymnasium_registry:
            del gymnasium_registry[env_id]
            print(f"Unregistered Gymnasium environment: {env_id}")
            return True
        print(f"Gymnasium environment '{env_id}' not found in registry.")
    except Exception:
        pass

    # Legacy OpenAI Gym fallback
    try:
        import gym
        if hasattr(gym.envs, "registry"):
            reg = gym.envs.registry
            # Older Gym: dict-like registry
            if isinstance(reg, dict) and env_id in reg:
                del reg[env_id]
                print(f"Unregistered legacy Gym environment: {env_id}")
                return True
            # Some versions expose env_specs
            if hasattr(reg, "env_specs") and env_id in reg.env_specs:
                del reg.env_specs[env_id]
                print(f"Unregistered legacy Gym environment (env_specs): {env_id}")
                return True
            print(f"Legacy Gym environment '{env_id}' not found in registry.")
    except Exception:
        pass

    return False

print(unregister_env('acc-variant-v1'))