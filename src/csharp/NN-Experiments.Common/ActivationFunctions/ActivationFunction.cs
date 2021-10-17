using System;

namespace NNExperiments.Common.ActivationFunctions
{
    /// <summary>
    /// Activation function. Contains direct and derivative methods (delegates).
    /// </summary>
    public struct ActivationFunction
    {
        public ActivationFunction(Func<double, double> func, Func<double, double> derivativeFunc, Range<double> inputRange, Range<double> outputRange)
        {
            Type = ActivationFunctionType.Custom;
            Calculate = func;
            CalculateDerivative = derivativeFunc;
            InputRange = inputRange;
            OutputRange = outputRange;
        }

        public ActivationFunction(ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            Type = activationFunctionType;
            // Sigmoid:
            Calculate = ActivationFunctions.Sigmoid;
            CalculateDerivative = ActivationFunctions.SigmoidDerivated;
            InputRange = new Range<double>(0, 1);
            OutputRange = new Range<double>(0, 1);
            switch (activationFunctionType)
            {
                case ActivationFunctionType.Sigmoid:
                    // Initiated above
                    break;
                case ActivationFunctionType.Tanh:
                    Calculate = ActivationFunctions.HyperbolicTangent;
                    CalculateDerivative = ActivationFunctions.HyperbolicTangentDerivated;
                    InputRange = new Range<double>(-1, 1);
                    OutputRange = new Range<double>(-1, 1);
                    break;
                case ActivationFunctionType.ReLU:
                    Calculate = ActivationFunctions.ReLU;
                    CalculateDerivative = ActivationFunctions.ReLUDerivated;
                    InputRange = new Range<double>(0, 1);
                    OutputRange = new Range<double>(0, 1);
                    break;
                case ActivationFunctionType.LReLU:
                    Calculate = ActivationFunctions.ReLU;
                    CalculateDerivative = ActivationFunctions.LReLUDerivated;
                    InputRange = new Range<double>(0, 1);
                    OutputRange = new Range<double>(0, 1);
                    break;
                case ActivationFunctionType.RandomLReLU:
                    break;
                case ActivationFunctionType.Custom:
                    break;
                default:
                    break;
            }
        }

        public Func<double, double> Calculate { get; private set; }

        public Func<double, double> CalculateDerivative { get; private set; }

        public ActivationFunctionType Type { get; private set; }

        /// <summary>
        /// Recomended range for input data.
        /// </summary>
        public Range<double> InputRange { get; private set; }

        /// <summary>
        /// Recomended range for output data.
        /// </summary>
        public Range<double> OutputRange { get; private set; }
    }
}
