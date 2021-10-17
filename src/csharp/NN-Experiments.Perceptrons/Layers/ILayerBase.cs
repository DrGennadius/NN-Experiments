using NNExperiments.Common.ActivationFunctions;
using NNExperiments.Perceptrons.Neurons;

namespace NNExperiments.Perceptrons.Layers
{
    public interface ILayerBase
    {
        Neuron[] Neurons { get; set; }

        double[] Output { get; }

        double[][] Weights { get; }

        ActivationFunction ActivationFunction { get; set; }

        double[] Forward(double[] input);

        double[] Forward(ILayerBase layer);

        void UpdateWeights(double alpha, double momentumRate);
    }
}
