import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from engine import ScoreEngine, numeric_matrix, train_with_holdout

class EngineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.rows = [{"hours": float(h), "prior": float(p), "score": float(10+2*h+0.5*p)}
                     for h,p in rng.uniform([0,20], [10,60], (150,2))]
    def model(self, **kwargs):
        return ScoreEngine(["hours","prior"]).fit(self.rows, target="score", **kwargs)
    def test_known_coefficients_and_prediction(self):
        model = self.model()
        coef = model.coefficients()
        self.assertAlmostEqual(coef["intercept"],10,places=8)
        self.assertAlmostEqual(coef["weights"]["hours"],2,places=8)
        self.assertAlmostEqual(coef["weights"]["prior"],0.5,places=8)
        self.assertAlmostEqual(model.predict([{"hours":3,"prior":40}])[0],36,places=8)
    def test_gradient_descent(self):
        gd = self.model(solver="gd")
        self.assertTrue(gd.training["converged"])
        np.testing.assert_allclose(gd.theta,self.model().theta,atol=1e-7)
    def test_ridge_solvers_agree(self):
        np.testing.assert_allclose(self.model(l2=0.3).theta,
            self.model(solver="gd",l2=0.3).theta,atol=1e-7)
    def test_no_holdout_preprocessing_leakage(self):
        model,report = train_with_holdout(self.rows,["hours","prior"],"score",seed=3)
        indices=np.random.default_rng(3).permutation(len(self.rows))
        train=[self.rows[i] for i in indices[30:]]
        np.testing.assert_allclose(model.means,numeric_matrix(train,model.features).mean(axis=0))
        self.assertEqual(report["split"]["test_rows"],30)
        self.assertLess(report["test"]["raw_predictions"]["mae"],1e-8)
        self.assertGreater(report["mean_baseline"]["mae"],1)
    def test_save_load(self):
        model=self.model()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model.json"
            model.save(path)
            loaded=ScoreEngine.load(path)
            np.testing.assert_array_equal(model.predict(self.rows),loaded.predict(self.rows))
            self.assertEqual(model.coefficients(),loaded.coefficients())
    def test_imputation_and_range_warning(self):
        model=self.model()
        detail=model.predict_details([{"hours":None,"prior":100}])[0]
        self.assertEqual(detail["imputed_features"],["hours"])
        self.assertEqual(detail["outside_training_range"],["prior"])
        self.assertAlmostEqual(detail["raw_score"],10+2*model.medians[0]+50)
    def test_clipping_retains_raw(self):
        detail=self.model().predict_details([{"hours":100,"prior":50}])[0]
        self.assertEqual(detail["predicted_score"],100)
        self.assertGreater(detail["raw_score"],100)
        self.assertTrue(detail["clipped"])
    def test_collinear_and_constant_features(self):
        rows=[{"x":i,"double":2*i,"constant":1,"y":10+3*i} for i in range(10)]
        model=ScoreEngine(["x","double","constant"]).fit(rows,target="y")
        self.assertLess(model.training["rank"],model.training["columns"])
        np.testing.assert_allclose(model.predict(rows),[r["y"] for r in rows],atol=1e-8)
    def test_invalid_features(self):
        for value in ["abc","nan","inf",True]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.model().predict([{"hours":value,"prior":50}])
        with self.assertRaises(ValueError): self.model().predict([{"hours":1}])
        with self.assertRaises(ValueError): ScoreEngine(["hours","hours"])
        with self.assertRaises(ValueError): ScoreEngine([])
        with self.assertRaises(ValueError): ScoreEngine(["score"]).fit(self.rows,"score")
        with self.assertRaises(ValueError):
            ScoreEngine(["hours"]).fit([{"hours":None,"score":50}]*3,"score")
        with self.assertRaises(ValueError): self.model(learning_rate=100,solver="gd")
    def test_invalid_targets(self):
        with self.assertRaises(ValueError): self.model().evaluate([])
        with self.assertRaises(ValueError):
            ScoreEngine(["x"]).fit([{"x":1,"y":101},{"x":2,"y":50}],"y")
        with self.assertRaises(ValueError):
            ScoreEngine(["x"]).fit([{"x":1,"y":None},{"x":2,"y":50}],"y")
    def test_corrupt_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bad.json"
            self.model().save(path)
            data=json.loads(path.read_text())
            data["scales"][0]=0
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): ScoreEngine.load(path)
    def test_custom_scale_and_constant_target(self):
        model=ScoreEngine(["x"],score_max=20).fit([{"x":1,"y":10},{"x":2,"y":10}],"y")
        self.assertIsNone(model.evaluate([{"x":1,"y":10}])["raw_predictions"]["r2"])
        self.assertAlmostEqual(model.predict([{"x":9}])[0],10)

if __name__=="__main__": unittest.main()
