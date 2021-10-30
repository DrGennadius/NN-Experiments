using NNExperiments.Common.ActivationFunctions;
using NNExperiments.Perceptrons.Common;

namespace NNExperiments.Perceptrons.Alternatives
{
    public interface IPerceptronOld : IBasicPerceptron
    {
        void TransferWeightsFrom(IPerceptronOld otherPerceptron);

        double[][][] GetWeights();

        void SetWeights(double[][][] weights);

        double[][] GetBiases();

        ActivationFunction GetActivationFunction();

        double GetMomentumRate();
    }
}
