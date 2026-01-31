def test_pid_increases_bid_when_underspending():
    agent = PIDBiddingAgent(target_spend_rate=0.5)
    initial_factor = agent.adjustment_factor
    
    # Simulate underspending (0.1 vs target 0.5)
    agent.update_controller(current_spend_rate=0.1)
    
    assert agent.adjustment_factor > initial_factor
    print("Test Passed: Agent increased bids to catch up to budget target.")