from imitation.policies.base import HardCodedPolicy
import 

class ExpertAgent(HardCodedPolicy):

    def __init__(self, env):
        super().__init__(env.observation_space, env.action_space)
        self.env = env
    
    def _choose_action(self, obs):

        agent_position = self.env.agent.get_position()
        agent_orientation = self.env.agent.get_orientation()
        goal_position = self.env.goal_pos
        goal_orientation = self.env.goal_orientation

        x_diff = goal_position[0] - agent_position[0]
        y_diff = goal_position[1] - agent_position[1]
        z_diff = goal_position[2] - agent_position[2]
        orientation_diff_z = goal_orientation[2] - agent_orientation[2]
        orientation_diff_z = min(orientation_diff_z, 2 * np.pi - orientation_diff_z)

        action = np.array([x_diff, y_diff, z_diff, orientation_diff_z], dtype=np.float64)
        # print(action)
        return action