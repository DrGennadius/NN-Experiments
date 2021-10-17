using NNExperiments.Common.Training;
using System;

namespace NNExperiments.Common
{
    public class CommonFunctions
    {
        /// <summary>
        /// Mean Square Error.
        /// </summary>
        /// <param name="realOutputs"></param>
        /// <param name="targetOutputs"></param>
        /// <returns></returns>
        public static double MSE(double[] realOutputs, double[] targetOutputs)
        {
            double error = 0;
            for (int i = 0; i < realOutputs.Length; i++)
            {
                double e = targetOutputs[i] - realOutputs[i];
                error += e * e;
            }
            return error / realOutputs.Length;
        }

        public static double MeanBatchMSE(double[,] realOutputs, double[,] targetOutputs)
        {
            double error = 0;
            for (int i = 0; i < realOutputs.GetLength(0); i++)
            {
                double[] realOutputsRow = new double[realOutputs.GetLength(1)];
                double[] goalOutputsRow = new double[targetOutputs.GetLength(1)];
                for (var c = 0; c < realOutputs.GetLength(1); c++)
                {
                    realOutputsRow[c] = realOutputs[i, c];
                }
                for (var c = 0; c < targetOutputs.GetLength(1); c++)
                {
                    goalOutputsRow[c] = targetOutputs[i, c];
                }
                error += MSE(realOutputsRow, goalOutputsRow);
            }
            return error / realOutputs.GetLength(0);
        }

        public static double GeneralError(double[,] realOutputs, double[,] targetOutputs)
        {
            double error = 0;
            for (int i = 0; i < realOutputs.GetLength(0); i++)
            {
                double[] realOutputsRow = new double[realOutputs.GetLength(1)];
                double[] goalOutputsRow = new double[targetOutputs.GetLength(1)];
                for (var c = 0; c < realOutputs.GetLength(1); c++)
                {
                    realOutputsRow[c] = realOutputs[i, c];
                }
                for (var c = 0; c < targetOutputs.GetLength(1); c++)
                {
                    goalOutputsRow[c] = targetOutputs[i, c];
                }
                error += IndividualError(realOutputsRow, goalOutputsRow);
            }
            return error;
        }

        public static double IndividualError(double[] realOutputs, double[] targetOutputs)
        {
            double error = 0;
            for (int i = 0; i < realOutputs.Length; i++)
            {
                error += Math.Pow(realOutputs[i] - targetOutputs[i], 2);
            }
            return error;
        }

        public static void GetMinMax(double[] values, out double min, out double max)
        {
            double cmin = values[0];
            double cmax = values[0];
            foreach (double value in values)
            {
                if (value > cmax)
                {
                    cmax = value;
                }
                if (value < cmin)
                {
                    cmin = value;
                }
            }
            min = cmin;
            max = cmax;
        }

        public static void GetMinMax(double[,] values, out double min, out double max)
        {
            GetMinMax(Flatten(values), out min, out max);
        }

        public static void GetMinMax(double[,] values, out Range<double> range)
        {
            GetMinMax(values, out double min, out double max);
            range = new Range<double>(min, max);
        }

        public static Range<double> GetMinMax(double[,] values)
        {
            GetMinMax(values, out double min, out double max);
            return new Range<double>(min, max);
        }

        public static IORanges GetMinMax(TrainData trainData)
        {
            return new IORanges(GetMinMax(trainData.Inputs), GetMinMax(trainData.Outputs));
        }

        /// <summary>
        /// Normalize value to range [0,1].
        /// </summary>
        /// <param name="val"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static double Normalize(double val, double min, double max)
        {
            return (val - min) / (max - min);
        }

        /// <summary>
        /// Normalize value to range [0,1].
        /// </summary>
        /// <param name="val"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        public static double Normalize(double val, Interval interval)
        {
            return (val - interval.Start.Value) / (interval.End.Value - interval.Start.Value);
        }

        /// <summary>
        /// Normalize values to range [0,1].
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double[] Normalize(double[] values)
        {
            double[] normalizedValues = new double[values.Length];
            GetMinMax(values, out double min, out double max);
            for (int i = 0; i < values.Length; i++)
            {
                normalizedValues[i] = Normalize(values[i], min, max);
            }
            return normalizedValues;
        }

        /// <summary>
        /// Normalize values to range [0,1].
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double[,] Normalize(double[,] values)
        {
            int size1 = values.GetLength(0);
            int size2 = values.GetLength(1);
            double[] flattenNormalizedValues = Normalize(Flatten(values));
            double[,] normalizedValues = new double[size1, size2];
            for (int i = 0; i < size1; i++)
            {
                for (int k = 0; k < size2; k++)
                {
                    normalizedValues[i, k] = flattenNormalizedValues[i * size2 + k];
                }
            }
            return normalizedValues;
        }

        /// <summary>
        /// Normalize values with scale to range [<paramref name="scaledRangeMin"/>,<paramref name="scaledRangeMax"/>].
        /// </summary>
        /// <param name="values"></param>
        /// <param name="scaledRangeMin"></param>
        /// <param name="scaledRangeMax"></param>
        /// <returns></returns>
        public static double[,] Normalize(double[,] values, double scaledRangeMin, double scaledRangeMax)
        {
            int size1 = values.GetLength(0);
            int size2 = values.GetLength(1);
            double[] flattenNormalizedValues = Scale(Flatten(values), scaledRangeMin, scaledRangeMax);
            double[,] normalizedValues = new double[size1, size2];
            for (int i = 0; i < size1; i++)
            {
                for (int k = 0; k < size2; k++)
                {
                    normalizedValues[i, k] = flattenNormalizedValues[i * size2 + k];
                }
            }
            return normalizedValues;
        }

        /// <summary>
        /// Scale value from range [<paramref name="rangeMin"/>,<paramref name="rangeMax"/>] to range [<paramref name="scaledRangeMin"/>,<paramref name="scaledRangeMax"/>].
        /// </summary>
        /// <param name="val">Value</param>
        /// <param name="rangeMin">Range min</param>
        /// <param name="rangeMax">Range max</param>
        /// <param name="scaledRangeMin">Scaled range min</param>
        /// <param name="scaledRangeMax">Scaled range max</param>
        /// <returns></returns>
        public static double Scale(double val, double rangeMin, double rangeMax, double scaledRangeMin, double scaledRangeMax)
        {
            return scaledRangeMin + ((val - rangeMin) * (scaledRangeMax - scaledRangeMin) / (rangeMax - rangeMin));
        }

        /// <summary>
        /// Scale value from source range to scaled range.
        /// </summary>
        /// <param name="val">Value</param>
        /// <param name="sourceInterval"></param>
        /// <param name="scaledInterval"></param>
        /// <returns></returns>
        public static double Scale(double val, Interval sourceInterval, Interval scaledInterval)
        {
            return scaledInterval.Start.Value +
                ((val - sourceInterval.Start.Value) * (scaledInterval.End.Value - scaledInterval.Start.Value)
                / (sourceInterval.End.Value - sourceInterval.Start.Value));
        }

        /// <summary>
        /// Scale value from source range to scaled range.
        /// </summary>
        /// <param name="val">Value</param>
        /// <param name="sourceRange"></param>
        /// <param name="scaledRange"></param>
        /// <returns></returns>
        public static double Scale(double val, Range<double> sourceRange, Range<double> scaledRange)
        {
            return scaledRange.Min + ((val - sourceRange.Min) * (scaledRange.Max - scaledRange.Min) / (sourceRange.Max - sourceRange.Min));
        }

        /// <summary>
        /// Normalize values with scale to range [<paramref name="scaledRangeMin"/>,<paramref name="scaledRangeMax"/>].
        /// </summary>
        /// <param name="values"></param>
        /// <param name="scaledRangeMin"></param>
        /// <param name="scaledRangeMax"></param>
        /// <returns></returns>
        public static double[] Scale(double[] values, double scaledRangeMin, double scaledRangeMax)
        {
            GetMinMax(values, out double rangeMin, out double rangeMax);
            return Scale(values, rangeMin, rangeMax, scaledRangeMin, scaledRangeMax);
        }

        /// <summary>
        /// Normalize values with scale to range [<paramref name="scaledRangeMin"/>,<paramref name="scaledRangeMax"/>].
        /// </summary>
        /// <param name="values"></param>
        /// <param name="forceRangeMin"></param>
        /// <param name="forceRangeMax"></param>
        /// <param name="scaledRangeMin"></param>
        /// <param name="scaledRangeMax"></param>
        /// <returns></returns>
        public static double[] Scale(double[] values, double forceRangeMin, double forceRangeMax, double scaledRangeMin, double scaledRangeMax)
        {
            double[] normalizedValues = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                normalizedValues[i] = Scale(values[i], forceRangeMin, forceRangeMax, scaledRangeMin, scaledRangeMax);
            }
            return normalizedValues;
        }

        /// <summary>
        /// Normalize values with scale to range [<paramref name="forceInterval"/>,<paramref name="scaledInterval"/>].
        /// </summary>
        /// <param name="values"></param>
        /// <param name="forceInterval"></param>
        /// <param name="scaledInterval"></param>
        /// <returns></returns>
        public static double[] Scale(double[] values, Interval forceInterval, Interval scaledInterval)
        {
            double[] normalizedValues = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                normalizedValues[i] = Scale(values[i], forceInterval.Start.Value, forceInterval.End.Value, scaledInterval.Start.Value, scaledInterval.End.Value);
            }
            return normalizedValues;
        }

        /// <summary>
        /// Normalize values with scale to range [<paramref name="scaledRangeMin"/>,<paramref name="scaledRangeMax"/>].
        /// </summary>
        /// <param name="values"></param>
        /// <param name="forceRange"></param>
        /// <param name="scaledRange"></param>
        /// <returns></returns>
        public static double[] Scale(double[] values, Range<double> forceRange, Range<double> scaledRange)
        {
            double[] normalizedValues = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                normalizedValues[i] = Scale(values[i], forceRange.Min, forceRange.Max, scaledRange.Min, scaledRange.Max);
            }
            return normalizedValues;
        }

        /// <summary>
        /// Normalize values into <see cref="TrainData"/> to range [0,1].
        /// </summary>
        /// <param name="trainData"></param>
        /// <returns></returns>
        public static TrainData NormalizeTrainData(TrainData trainData)
        {
            trainData.Set(Normalize(trainData.Inputs), Normalize(trainData.Outputs));
            return trainData;
        }

        /// <summary>
        /// Create new <see cref="TrainData"/> instance with normalized values into origin <see cref="TrainData"/> to range [0,1].
        /// </summary>
        /// <param name="trainData"></param>
        /// <returns></returns>
        public static TrainData CreateNewNormalizeTrainData(TrainData trainData)
        {
            TrainData newTrainData = new(Normalize(trainData.Inputs), Normalize(trainData.Outputs));
            return newTrainData;
        }

        /// <summary>
        /// Create new <see cref="TrainData"/> instance with normalized values into origin <see cref="TrainData"/> with scale range for separate input and output.
        /// </summary>
        /// <param name="trainData"></param>
        /// <param name="scaledRangeInputMin"></param>
        /// <param name="scaledRangeInputMax"></param>
        /// <param name="scaledRangeOutputMin"></param>
        /// <param name="scaledRangeOutputMax"></param>
        /// <returns></returns>
        public static TrainData CreateNewNormalizeTrainData(TrainData trainData,
            double scaledRangeInputMin, double scaledRangeInputMax,
            double scaledRangeOutputMin, double scaledRangeOutputMax)
        {
            return new TrainData(
                Normalize(trainData.Inputs, scaledRangeInputMin, scaledRangeInputMax),
                Normalize(trainData.Outputs, scaledRangeOutputMin, scaledRangeOutputMax)
            );
        }

        /// <summary>
        /// Create new <see cref="TrainData"/> instance with normalized values into origin <see cref="TrainData"/> with scale range for input and output.
        /// </summary>
        /// <param name="trainData"></param>
        /// <param name="scaledRangeMin"></param>
        /// <param name="scaledRangeMax"></param>
        /// <returns></returns>
        public static TrainData CreateNewNormalizeTrainData(TrainData trainData, double scaledRangeMin, double scaledRangeMax)
        {
            return CreateNewNormalizeTrainData(trainData, scaledRangeMin, scaledRangeMax, scaledRangeMin, scaledRangeMax);
        }

        public static TrainData CreateNewNormalizeTrainData(TrainData trainData, Range<double> inputRange, Range<double> outputRange)
        {
            return CreateNewNormalizeTrainData(trainData, inputRange.Min, inputRange.Max, outputRange.Min, outputRange.Max);
        }

        public static double[] Flatten(double[,] values)
        {
            int size1 = values.GetLength(0);
            int size2 = values.GetLength(1);
            double[] result = new double[size1 * size2];
            int write = 0;
            for (int i = 0; i < size1; i++)
            {
                for (int z = 0; z < size2; z++)
                {
                    result[write++] = values[i, z];
                }
            }
            return result;
        }
    }
}
