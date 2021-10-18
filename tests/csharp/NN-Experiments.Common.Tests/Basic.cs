using NNExperiments.Common;
using NUnit.Framework;
using System.Linq;

namespace NNExperiments.Common.Tests
{
    public class Basic
    {
        private readonly double[][] _testData1 = new double[][]
        {
            new double[] { 0, 1, 2 },
            new double[] { 0, 0.5, 1 },
            new double[] { -1, 0, 1 }
        };

        private readonly double[][] _scaleRanges = new double[][]
        {
            new double[] { 0, 1 },
            new double[] { -1, 1 },
            new double[] { -1, 0 },
            new double[] { -100, 100 }
        };

        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void NormalizeAndScaleTest()
        {
            double[][][] scaledData = new double[3][][];

            for (int i = 0; i < 3; i++)
            {
                double[] normalizedData = CommonFunctions.Normalize(_testData1[i]);
                Assert.AreEqual(normalizedData.Min(), 0);
                Assert.AreEqual(normalizedData.Max(), 1);

                scaledData[i] = NormalizeAndScaleSubTest1(_testData1[i]);
            }

            scaledData = scaledData.Select(x => x.Reverse().ToArray()).ToArray();
            for (int i = 0; i < 3; i++)
            {
                for (int k = 0; k < _scaleRanges.Length; k++)
                {
                    ScaleDataSubTest1(scaledData[i][k], _scaleRanges[i]);
                }
            }

            Assert.Pass();
        }

        private double[][] NormalizeAndScaleSubTest1(double[] data)
        {
            double[][] result = new double[_scaleRanges.Length][];

            for (int i = 0; i < _scaleRanges.Length; i++)
            {
                result[i] = ScaleDataSubTest1(data, _scaleRanges[i]);
            }

            return result;
        }

        private double[] ScaleDataSubTest1(double[] data, double[] scaleRange)
        {
            double[] scaledData = CommonFunctions.Scale(data, scaleRange[0], scaleRange[1]);
            Assert.AreEqual(scaledData.Min(), scaleRange[0]);
            Assert.AreEqual(scaledData.Max(), scaleRange[1]);
            Assert.AreEqual(scaledData[1], (scaleRange[0] + scaleRange[1]) / 2);

            return scaledData;
        }
    }
}