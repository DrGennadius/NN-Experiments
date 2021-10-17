using NNExperiments.Common.ActivationFunctions;

namespace NNExperiments.Perceptrons.Layers
{
    public struct LayerStruct
    {
        public double[,] Weights;
        public double[,] PrevWeightChanges;
        public double[] Biases;
        public double[] PrevBiasChanges;
        public double[] Outputs;
        public double[] Derivatives;
        public double[] Deltas;

        public ActivationFunction ActivationFunction;

        public double[] Forward(double[] input)
        {
            double[] result = new double[Weights.GetLength(0)];
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                result[i] = ForwardNeuron(i, input);
            }
            Outputs = result;
            return result;
        }

        public double ForwardNeuron(int neuronIndex, double[] input)
        {
            double sum = Biases[neuronIndex];

            for (int i = 0; i < Weights.GetLength(1); i++)
            {
                sum += Weights[neuronIndex, i] * input[i];
            }

            double result = ActivationFunction.Calculate(sum);
            Outputs[neuronIndex] = result;
            Derivatives[neuronIndex] = ActivationFunction.CalculateDerivative(result);
            return result;
        }
    }
}
