from agents.tabular_q import TabularQAgent

def run_dqn(*args, **kwargs):
    from agents.dqn import run_dqn as _run_dqn
    return _run_dqn(*args, **kwargs)
