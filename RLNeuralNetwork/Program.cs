using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace RLNeuralNetwork
{
    internal class Program
    {

        internal struct InputData
        {
            public InputData(double[] inputs, double value)
            {
                this.inputs = inputs;
                this.value = value;
            }

            public double[] inputs;
            public double value;
        }

        static InputData[] ReadOverwatchResults(string filePath)
        {
            StreamReader sr = new StreamReader(filePath);
            string arrayString = sr.ReadToEnd();
            arrayString = arrayString.Replace("\t", "").Replace(";","");
            string[] arraysByMatch = arrayString.Split("Global.inputResults = ");
            List<InputData> inputDatas = new List<InputData>();
            for (int i = 1; i < arraysByMatch.Length; i++)
            {
                arraysByMatch[i] = arraysByMatch[i].Substring(6);
                if (i + 1 == arraysByMatch.Length)
                {
                    arraysByMatch[i] = arraysByMatch[i].Remove(arraysByMatch[i].Length - 1);
                } 
                else
                {
                    arraysByMatch[i] = arraysByMatch[i].Remove(arraysByMatch[i].Length - 3);
                }

                arraysByMatch[i] = arraysByMatch[i].Replace("\r\n", "");
                arraysByMatch[i] = arraysByMatch[i].Replace(" ", "");

                string[] dataStrings = arraysByMatch[i].Split("Array(");

                for (int j = 1; j < dataStrings.Length; j++)
                {
                    string inputsString = dataStrings[j].Split("),")[0];
                    string[] inputsStrings = inputsString.Split(',');
                    string valueString = dataStrings[j].Split("),")[1];
                    valueString = valueString.Replace(",", "");

                    List<double> inputs = new List<double>();
                    for (int k = 0; k < inputsStrings.Length; k++)
                    {
                        double input = double.Parse(inputsStrings[k]);
                        if (k >= 49 && k <= 51)
                        {
                            if (input < 0)
                            {
                                input = -0;
                            }
                        }
                        inputs.Add(input);
                    }
                    double value = double.Parse(valueString);
                    inputDatas.Add(new InputData(inputs.ToArray(), value));
                }
                
            }
            return inputDatas.ToArray();
        }

        void ToOverwatchArray()
        {

        }

        static InputData[] read()
        {
            Console.WriteLine("Enter filepath:");
            string path = Console.ReadLine();
            InputData[] data = ReadOverwatchResults(path);
            return data;
        }
        
        void test()
        {
            List<HiddenLayer> layers = new List<HiddenLayer>();
            //layers.Add(new HiddenLayer(new double[,] { { 0.3, 0.6, -0.4 }, { 0.8, -0.7, -0.5 } }));
            layers.Add(new HiddenLayer(new double[,] { { 0.3, 0.8 }, { 0.6, -0.7 }, { -0.4, -0.5 } }, 0.1));

            NeuralNetwork NN = new NeuralNetwork(layers.ToArray(), new OutputLayer(new double[,] { { -0.3 }, { 0.2 } }, 0.01));
            Console.WriteLine(NN.feedForward(new double[] { 0.36, -1, 0.4 })[0]);
            NN.backPropagate(new double[] { 0.36, -1, 0.4 }, new double[] { 5 }, 1);
            Console.WriteLine(NN.optimisedFeedForward(new double[] { 0.36, -1 }, new double[,] { { 0.4 } })[0, 0]);

            for (int i = 0; i < 1000; i++)
            {
                //NN.backPropagate(new double[] { 0.36, -1, 0.4 }, new double[] { 5 });
                //Console.WriteLine(NN.feedForward(new double[] { 0.36, -1, 0.4 })[0]);
            }
        }

        void test2()
        {
            NeuralNetwork NN = new NeuralNetwork(new HiddenLayer[] { new HiddenLayer(1000, 800, 0.1) }, new OutputLayer(800, 1, 0.01));
            double[] test = randomDoubleArray(1000);
            double[] test2 = randomDoubleArray(1000);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            double result = NN.feedForward(test)[0];
            sw.Stop();
            Console.WriteLine(result);
            for (int i = 0; i < 100; i++)
            {
                NN.backPropagate(test, new double[] { 500 }, 1);
                NN.backPropagate(test2, new double[] { -700 }, 1);
            }
            result = NN.feedForward(test)[0];
            Console.WriteLine(result);
            result = NN.feedForward(test2)[0];
            Console.WriteLine(result);
            Console.WriteLine(sw.ElapsedTicks);
        }

        static void testLoss(InputData[] data)
        {
            double metaLoss = 0;
            double metaLossCount = 0;
            double metaLoss2 = 0;
            double metaLossCount2 = 0;
            for (int i = 0; i < 5; i++)
            {
                NeuralNetwork NN = new NeuralNetwork(new HiddenLayer[] { new HiddenLayer(55, 50, 0.0001), new HiddenLayer(50, 40, 0.0001) }, new OutputLayer(40, 1, 0.0001));
                for (int j = 0; j < 50; j++)
                {
                    foreach (InputData x in data)
                    {
                        NN.backPropagate(x.inputs, new double[] { x.value }, 1 / ((i * 10) + 1));
                    }
                }
                double loss = 0;
                int lossCount = 0;
                double loss2 = 0;
                int lossCount2 = 0;
                foreach (InputData x in data)
                {
                    if (x.inputs[49] == 0.1 || x.inputs[50] == 0.1 || x.inputs[51] == 0.1)
                    {
                        loss2 += Math.Abs(x.value - NN.feedForward(x.inputs)[0]);
                        lossCount2++;
                    }
                    else
                    {
                        loss += Math.Abs(x.value - NN.feedForward(x.inputs)[0]);
                        lossCount++;
                    }
                }
                metaLoss += loss / lossCount;
                metaLossCount++;
                metaLoss2 += loss2 / lossCount2;
                metaLossCount2++;
            }
            Console.WriteLine(metaLoss / metaLossCount);
            Console.WriteLine(metaLoss2 / metaLossCount2);
            Console.WriteLine((metaLoss + metaLoss2) / (metaLossCount + metaLossCount2));
            Console.WriteLine("ee");
        }

        static double[] randomDoubleArray(int length)
        {
            Random random = new Random();
            double[] array = new double[length];
            for (int i = 0; i < length; i++)
            {
                array[i] = random.NextDouble() * 0.0005;
            }
            return array;
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Enter filepath:");
            string path = Console.ReadLine();
            InputData[] data = ReadOverwatchResults(path);
            testLoss(data);
            Console.WriteLine("-------------------");
            double metaLoss = 0;
            double metaLossCount = 0;
            double metaLoss2 = 0;
            double metaLossCount2 = 0;
            for (int i = 0; i < 5; i++)
            {
                WideDeepNeuralNetwork WDNN = new WideDeepNeuralNetwork(new HiddenLayer[] { new HiddenLayer(55, 50, 0.0001), new HiddenLayer(50, 40, 0.0001) }, new WideDeepOutputLayer(40, 55, 1, 0.00001, 0.00001));
                for (int j = 0; j < 50; j++)
                {
                    foreach (InputData x in data)
                    {
                        WDNN.backPropagate(x.inputs, x.inputs, new double[] { x.value }, 1 / ((i * 10) + 1));
                    }
                }
                double loss = 0;
                int lossCount = 0;
                double loss2 = 0;
                int lossCount2 = 0;
                foreach (InputData x in data)
                {
                    if (x.inputs[49] == 0.1 || x.inputs[50] == 0.1 || x.inputs[51] == 0.1)
                    {
                        loss2 += Math.Abs(x.value - WDNN.wideDeepFeedForward(x.inputs, x.inputs)[0]);
                        lossCount2++;
                    }
                    else
                    {
                        loss += Math.Abs(x.value - WDNN.wideDeepFeedForward(x.inputs, x.inputs)[0]);
                        lossCount++;
                    }
                }
                metaLoss += loss / lossCount;
                metaLossCount++;
                metaLoss2 += loss2 / lossCount2;
                metaLossCount2++;
            }
            Console.WriteLine(metaLoss / metaLossCount);
            Console.WriteLine(metaLoss2 / metaLossCount2);
            Console.WriteLine((metaLoss + metaLoss2) / (metaLossCount + metaLossCount2));
            Console.WriteLine("ee");
        }
    }
}