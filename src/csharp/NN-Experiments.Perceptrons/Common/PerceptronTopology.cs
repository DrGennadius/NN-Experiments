using NNExperiments.Common.ActivationFunctions;

namespace NNExperiments.Perceptrons.Common
{
    public struct PerceptronTopology
    {
        public PerceptronTopology(int inputSize, int[] neuronsPerHiddenLayer, int outputSize, ActivationFunction activationFunction)
        {
            InputSize = inputSize;
            HiddenLayerSizes = neuronsPerHiddenLayer;
            OutputSize = outputSize;
            ActivationFunction[] activationFunctions = new ActivationFunction[HiddenLayerSizes.Length + 1];
            for (int i = 0; i < activationFunctions.Length; i++)
            {
                activationFunctions[i] = activationFunction;
            }
            ActivationFunctions = activationFunctions;
        }

        public PerceptronTopology(int inputSize, int[] neuronsPerLayer, ActivationFunction activationFunction)
        {
            InputSize = inputSize;
            OutputSize = neuronsPerLayer[^1];
            HiddenLayerSizes = new int[neuronsPerLayer.Length - 1];
            for (int i = 0; i < neuronsPerLayer.Length - 1; i++)
            {
                HiddenLayerSizes[i] = neuronsPerLayer[i];
            }
            ActivationFunction[] activationFunctions = new ActivationFunction[HiddenLayerSizes.Length + 1];
            for (int i = 0; i < activationFunctions.Length; i++)
            {
                activationFunctions[i] = activationFunction;
            }
            ActivationFunctions = activationFunctions;
        }

        public PerceptronTopology(int[] sizes, ActivationFunction activationFunction)
        {
            InputSize = sizes[0];
            OutputSize = sizes[^1];
            HiddenLayerSizes = new int[sizes.Length - 2];
            for (int i = 1; i < sizes.Length - 1; i++)
            {
                HiddenLayerSizes[i - 1] = sizes[i];
            }
            ActivationFunction[] activationFunctions = new ActivationFunction[sizes.Length - 1];
            for (int i = 0; i < activationFunctions.Length; i++)
            {
                activationFunctions[i] = activationFunction;
            }
            ActivationFunctions = activationFunctions;
        }

        public PerceptronTopology(int inputSize, int[] neuronsPerHiddenLayer, int outputSize, ActivationFunction[] activationFunctions)
        {
            InputSize = inputSize;
            HiddenLayerSizes = neuronsPerHiddenLayer;
            OutputSize = outputSize;
            ActivationFunctions = activationFunctions;
        }

        public PerceptronTopology(int inputSize, int[] neuronsPerLayer, ActivationFunction[] activationFunctions)
        {
            InputSize = inputSize;
            OutputSize = neuronsPerLayer[^1];
            HiddenLayerSizes = new int[neuronsPerLayer.Length - 1];
            for (int i = 0; i < neuronsPerLayer.Length - 1; i++)
            {
                HiddenLayerSizes[i] = neuronsPerLayer[i];
            }
            ActivationFunctions = activationFunctions;
        }

        public PerceptronTopology(int[] sizes, ActivationFunction[] activationFunctions)
        {
            InputSize = sizes[0];
            OutputSize = sizes[^1];
            if (sizes.Length == 2)
            {
                HiddenLayerSizes = System.Array.Empty<int>();
            }
            else
            {
                HiddenLayerSizes = new int[sizes.Length - 2];
                for (int i = 1; i < sizes.Length - 1; i++)
                {
                    HiddenLayerSizes[i - 1] = sizes[i];
                }
            }
            ActivationFunctions = activationFunctions;
        }

        public int InputSize;

        public int[] HiddenLayerSizes;

        public int OutputSize;

        public ActivationFunction[] ActivationFunctions { get; private set; }

        /// <summary>
        /// Input size + neurons per hidden layer + output size.
        /// </summary>
        public int[] GetSizes()
        {
            int[] sizes = new int[HiddenLayerSizes.Length + 2];
            sizes[0] = InputSize;
            for (int i = 0; i < HiddenLayerSizes.Length; i++)
            {
                sizes[i + 1] = HiddenLayerSizes[i];
            }
            sizes[^1] = OutputSize;
            return sizes;
        }

        public int GetLayerCount()
        {
            return HiddenLayerSizes.Length + 1;
        }

        /// <summary>
        /// Get the activation function for first layer (input).
        /// </summary>
        /// <returns></returns>
        public ActivationFunction GetInputActivationFunction()
        {
            return ActivationFunctions[0];
        }

        /// <summary>
        /// Get the activation function for last layer (output).
        /// </summary>
        /// <returns></returns>
        public ActivationFunction GetOutputActivationFunction()
        {
            return ActivationFunctions[^1];
        }
    }
}
