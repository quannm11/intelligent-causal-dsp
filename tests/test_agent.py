import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uplift_project.src.agents.agent import PIDBiddingAgent

def run_test():
    # Target 20% spend
    agent = PIDBiddingAgent(kp=0.1, ki=0.01, kd=0.05, target_spend_rate=0.2)
    
    print(f"Initial Adjustment Factor: {agent.adjustment_factor}")
    
    # Simulate underspending (only 5% spent)
    agent.update_controller(current_spend_rate=0.05)
    print(f"Factor after Underspend: {agent.adjustment_factor:.4f} (Should be > 1.0)")
    
    # Simulate overspending (80% spent)
    agent.update_controller(current_spend_rate=0.80)
    print(f"Factor after Overspend: {agent.adjustment_factor:.4f} (Should have decreased)")

if __name__ == "__main__":
    run_test()