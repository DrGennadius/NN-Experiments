using NNExperiments.Common.ActivationFunctions;

namespace NNExperiments.Perceptrons.Neurons
{
    public interface INeuron
    {
        double[] Weights { get; set; }
        double[] Input { get; set; }
        double[] PreviousChanges { get; set; }
        double Bias { get; set; }
        double Output { get; set; }
        double DerivatedOutput { get; set; }
        double PreviousBiasChange { get; set; }
        double Delta { get; set; }

        double Forward(double[] input);
        double Forward(ActivationFunction activationFunction, double[] input);
    }
}
