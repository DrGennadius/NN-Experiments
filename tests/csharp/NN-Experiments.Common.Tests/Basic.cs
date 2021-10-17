using NNExperiments.Common;
using NUnit.Framework;

namespace NNExperiments.Common.Tests
{
    public class Basic
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void NormalizeAndScaleTest()
        {
            double[][] data = new double[][]
            {
                new double[] { 0, 1, 2 },
                new double[] { 0, 0.5, 1 },
                new double[] { -1, 0, 1 }
            };

            double[][] scaledData0 = new double[3][];
            double[][] scaledData1 = new double[3][];
            double[][] scaledData2 = new double[3][];
            double[][] scaledData3 = new double[3][];
            double[][] scaledData4 = new double[3][];

            for (int i = 0; i < 3; i++)
            {
                scaledData0[i] = CommonFunctions.Normalize(data[i]);
                scaledData1[i] = CommonFunctions.Scale(data[i], 0, 1);
                scaledData2[i] = CommonFunctions.Scale(data[i], -1, 1);
                scaledData3[i] = CommonFunctions.Scale(data[i], -1, 0);
                scaledData4[i] = CommonFunctions.Scale(data[i], -100, 100);
            }

            double[][] scaledData5 = new double[3][];
            double[][] scaledData6 = new double[3][];
            double[][] scaledData7 = new double[3][];
            double[][] scaledData8 = new double[3][];
            double[][] scaledData9 = new double[3][];

            for (int i = 0; i < 3; i++)
            {
                scaledData5[i] = CommonFunctions.Normalize(scaledData4[i]);
                scaledData6[i] = CommonFunctions.Scale(scaledData3[i], 0, 1);
                scaledData7[i] = CommonFunctions.Scale(scaledData2[i], -1, 1);
                scaledData8[i] = CommonFunctions.Scale(scaledData1[i], -1, 0);
                scaledData9[i] = CommonFunctions.Scale(scaledData0[i], -100, 100);
            }

            Assert.Pass();
        }
    }
}