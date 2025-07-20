using Microsoft.ML.Data;
public class SentimentPrediction : SentimentData
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    public float Score { get; set; }
    public float Probability { get; set; }
}