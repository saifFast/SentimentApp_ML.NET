using Microsoft.ML;

public class SentimentService
{
    private readonly MLContext _mlContext;
    private readonly PredictionEngine<SentimentData, SentimentPrediction> _predEngine;

    public SentimentService(string dataPath)
    {
        _mlContext = new MLContext();
        var dataView = LoadData(dataPath);
        var model = BuildAndTrainModel(dataView);
        _predEngine = _mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
    }

    public SentimentPrediction Predict(string text)
    {
        var input = new SentimentData { Text = text };
        return _predEngine.Predict(input);
    }

    private IDataView LoadData(string dataPath)
    {
        return _mlContext.Data.LoadFromTextFile<SentimentData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ',');
    }

    private ITransformer BuildAndTrainModel(IDataView dataView)
    {
        var pipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
            .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: nameof(SentimentData.Label), featureColumnName: "Features"));

        return pipeline.Fit(dataView);
    }
}