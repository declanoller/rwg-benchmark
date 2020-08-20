import path_utils
import os
import Sample

best_agent_dir = ""

dir = os.path.join(path_utils.get_output_dir(), best_agent_dir)

s = Sample.load_best_agent_sample_from_dir(dir)

s.run_episode(show_ep=True)
