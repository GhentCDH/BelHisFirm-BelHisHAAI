from CRF.predict_crf import Predict

if __name__ == "__main__":
    model_path = "/home/bas/Documents/Visual Code Repos/BelHisFirm-BelHisHAAI/src/index_parser/model/CRF_1884.pkg"
    predictor = Predict(model_path=model_path)
    test_line = "2767. Schütz et Diden, à Anvers. — Retraite d'associés."
    predictor.predict_single_line(test_line, debug=True)
    output = predictor.get_output_no_punctuation()
    print(output)