import path_utils
import os
import Sample


dir = os.path.join(
    path_utils.get_output_dir(), "Pendulum-v0_sample_09-07-2020_01-44-48"
)

s = Sample.load_best_agent_sample_from_dir(dir)

s.run_episode(show_ep=True)
