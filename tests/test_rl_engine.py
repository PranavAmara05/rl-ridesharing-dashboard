import unittest
import numpy as np
from ridesharing_darm_dprs_dqn import (
    G, Req, Veh, insert_req, customer_decide, 
    price_initial, price_driver, 
    MAX_DETOUR_FACTOR, REJECT_RADIUS_KM
)

class TestRLEngine(unittest.TestCase):

    def test_city_grid_utilities(self):
        # Test zid (zone id)
        self.assertEqual(G.zid(0, 0), 0)
        self.assertEqual(G.zid(G.H-1, G.W-1), G.n-1)
        # Test rc (row, col)
        r, c = G.rc(0)
        self.assertEqual(r, 0); self.assertEqual(c, 0)
        # Test dist_km
        d = G.dist_km(0, 1) # Adjacent cells
        self.assertAlmostEqual(d, 0.8) # ZONE_KM = 0.8

    def test_pricing_logic(self):
        v = Veh(vid=1, vt=0, zone=0, cap=0)
        req = Req(o=0, d=1, np=1, t=0)
        
        # Initial price check
        p_init = price_initial(v, req, cost_km=1.0, wait_min=2.0)
        self.assertGreater(p_init, 0)
        
        # Driver price markup for bad zones
        # Mock hotspot_zones as empty to force markup
        p_driver = price_driver(v, req, p_init, hotspot_zones=[], zone_rank={1: 224})
        self.assertGreater(p_driver, p_init)

    def test_customer_decision(self):
        v = Veh(vid=1, vt=0, zone=0, cap=0)
        req = Req(o=0, d=1, np=1, t=0)
        req.delta = 100.0 # Very high delta = always accept
        
        accepted, reason = customer_decide(req, v, price=10.0, wait_min=1.0)
        self.assertTrue(accepted)
        self.assertIsNone(reason)
        
        req.delta = -100.0 # Very low delta = always reject
        accepted, reason = customer_decide(req, v, price=10.0, wait_min=1.0)
        self.assertFalse(accepted)
        self.assertIn(reason, ['price', 'wait'])

    def test_route_insertion(self):
        v = Veh(vid=1, vt=0, zone=0, cap=0)
        req = Req(o=0, d=1, np=1, t=0)
        
        # First insertion into empty route
        cost, extra, route, wait = insert_req(v, req, t=0)
        self.assertIsNotNone(route)
        self.assertEqual(len(route), 2) # ('pu', req), ('do', req)
        self.assertLess(extra, float('inf'))

        # Test capacity constraint
        v.cap = v.maxcap # Full
        cost, extra, route, wait = insert_req(v, req, t=0)
        self.assertIsNone(route)

if __name__ == "__main__":
    unittest.main()
