using System;
using System.Diagnostics;
using System.IO;
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
                                input = -0.1;
                            }
                            //input *= 10;
                        }
                        inputs.Add(input);
                    }
                    double value = double.Parse(valueString);
                    inputDatas.Add(new InputData(inputs.ToArray(), value));
                }
                
            }
            return inputDatas.ToArray();
        }

        static void ExportAsOverwatchArray(double[,] w1, double[] w2, double[] wideW)
        {
            const int dp = 5;
            StreamWriter sw = new("export.txt");
            sw.WriteLine("actions");
            sw.WriteLine("{");
            string weights = "\tw1Import1 = Array(";
            int import1length = (int)(w1.GetLength(0) / 1.5);
            for (int i = 0; i < import1length; i++)
            {
                weights += "Array(";
                for (int j = 0; j < w1.GetLength(1); j++)
                {
                    weights += j != w1.GetLength(1) - 1 ? Math.Round(w1[i, j], dp) + ", " : Math.Round(w1[i, j], dp);
                }
                weights += i != import1length - 1 ? "), " : "));";
            }
            sw.WriteLine(weights);

            sw.WriteLine("}");
            sw.WriteLine("actions");
            sw.WriteLine("{");

            weights = "\tw1Import2 = Array(";
            for (int i = import1length; i < w1.GetLength(0); i++)
            {
                weights += "Array(";
                for (int j = 0; j < w1.GetLength(1); j++)
                {
                    weights += j != w1.GetLength(1) - 1 ? Math.Round(w1[i, j], dp) + ", " : Math.Round(w1[i, j], dp);
                }
                weights += i != w1.GetLength(0) - 1 ? "), " : "));";
            }
            sw.WriteLine(weights);

            weights = "\tw2Import = Array(";
            for (int i = 0; i < w2.Length; i++)
            {
                weights += i != w2.Length - 1 ? Math.Round(w2[i], dp) + ", " : Math.Round(w2[i], dp) + ");";
            }
            sw.WriteLine(weights);

            weights = "\twideWeightImport = Array(";
            for (int i = 0; i < wideW.Length; i++)
            {
                weights += i != wideW.Length - 1 ? Math.Round(wideW[i], dp) + ", " : Math.Round(wideW[i], dp) + ");";
            }
            sw.WriteLine(weights);

            sw.WriteLine("}");
            sw.Close();
            Console.WriteLine("weights exported");
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

        static void testLoss(InputData[] data, InputData[] wideData)
        {
            double metaLoss = 0;
            double metaLossCount = 0;
            double metaLoss2 = 0;
            double metaLossCount2 = 0;
            for (int i = 0; i < 10; i++)
            {
                NeuralNetwork NN = new NeuralNetwork(new HiddenLayer[] { new HiddenLayer(55, 55, 0.01) }, new OutputLayer(55, 1, 0.01));
                for (int j = 1; j <= 2; j++)
                {
                    foreach (InputData x in data)
                    {
                        NN.backPropagate(x.inputs, new double[] { x.value }, 1 / ((j)));
                    }
                }
                double loss = 0;
                int lossCount = 0;
                double loss2 = 0;
                int lossCount2 = 0;
                foreach (InputData x in data)
                {
                    if (x.inputs[49] >= 0.01 || x.inputs[50] >= 0.01 || x.inputs[51] >= 0.01)
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
            Console.WriteLine("-------------------");
            metaLoss = 0;
            metaLossCount = 0;
            metaLoss2 = 0;
            metaLossCount2 = 0;
            WideDeepNeuralNetwork s;
            for (int i = 0; i < 10; i++)
            {
                WideDeepNeuralNetwork WDNN = new WideDeepNeuralNetwork(new HiddenLayer[] { new HiddenLayer(55, 55, 0.01) }, new WideDeepOutputLayer(55, 55, 1, 0.001, 0.00001));
                for (int j = 1; j <= 2; j++)
                {
                    for (int k = 0; k < data.Length; k++)
                    {
                        WDNN.backPropagate(data[k].inputs, wideData[k].inputs, new double[] { data[k].value }, 1 / ((j)));
                    }
                }
                double loss = 0;
                int lossCount = 0;
                double loss2 = 0;
                int lossCount2 = 0;
                for (int k = 0; k < data.Length; k++)
                {
                    if (data[k].inputs[49] >= 0.01 || data[k].inputs[50] >= 0.01 || data[k].inputs[51] >= 0.01)
                    {
                        loss2 += Math.Abs(data[k].value - WDNN.wideDeepFeedForward(data[k].inputs, wideData[k].inputs)[0]);
                        lossCount2++;
                    }
                    else
                    {
                        loss += Math.Abs(data[k].value - WDNN.wideDeepFeedForward(data[k].inputs, wideData[k].inputs)[0]);
                        lossCount++;
                    }
                }
                metaLoss += loss / lossCount;
                metaLossCount++;
                metaLoss2 += loss2 / lossCount2;
                metaLossCount2++;
                s = WDNN;
            }
            Console.WriteLine(metaLoss / metaLossCount);
            Console.WriteLine(metaLoss2 / metaLossCount2);
            Console.WriteLine((metaLoss + metaLoss2) / (metaLossCount + metaLossCount2));
            Console.WriteLine("ee"); //literally just for breakpoints
        }

        static void createOWArray(InputData[] data, InputData[] wideData)
        {
            WideDeepNeuralNetwork WDNN = new WideDeepNeuralNetwork(new HiddenLayer[] { new HiddenLayer(55, 55, 0.01) }, new WideDeepOutputLayer(55, 55, 1, 0.001, 0.00001));
            for (int i = 1; i <= 2; i++)
            {
                for (int j = 0; j < data.Length; j++)
                {
                    WDNN.backPropagate(data[j].inputs, wideData[j].inputs, new double[] { data[j].value }, 1 / ((i)));
                }
            }
            double[] w1 = new double[WDNN.outputLayer.weights.GetLength(0)];
            double[] ww = new double[WDNN.outputLayer.weights.GetLength(0)];
            for (int i = 0; i < WDNN.outputLayer.weights.GetLength(0); i++)
            {
                w1[i] = WDNN.outputLayer.weights[i, 0];
                ww[i] = ((WideDeepOutputLayer)WDNN.outputLayer).wideWeights[i, 0];
            }

            ExportAsOverwatchArray(WDNN.hiddenLayers[0].weights, w1, ww);
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
            InputData[] wideData = data;
            for (int i = 49; i <= 51; i++)
            {
                for (int j = 0; j < wideData.Length; j++)
                {
                    if (wideData[j].inputs[i] < 0)
                    {
                        //wideData[j].inputs[i] = 0;
                    }
                }
            }
            //testLoss(data, wideData);
            createOWArray(data, wideData);
        }
    }
}