﻿using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

namespace RLNeuralNetwork
{
    internal class Program
    {

        internal struct InputData
        {
            public InputData(double[] inputs, double[] values)
            {
                this.inputs = inputs;
                this.values = values;
            }

            public double[] inputs;
            public double[] values;
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
                    inputDatas.Add(new InputData(inputs.ToArray(), new double[] {value}));
                }
                
            }
            return inputDatas.ToArray();
        }

        static InputData[] ReadAdvancedOverwatchResults(string filePath)
        {
            StreamReader sr = new StreamReader(filePath);
            string arrayString = sr.ReadToEnd();
            arrayString = arrayString.Replace("\t", "").Replace(";", "").Replace("\r\n", "").Replace(" ", "");
            string[] arraysByMatch = arrayString.Split("Global.inputResults=Array(");
            List<InputData> inputDatas = new List<InputData>();
            //get each set of results/match, then iterate over them.
            for (int i = 1; i < arraysByMatch.Length; i++)
            {
                //Get each action result, then iterate over them
                string[] results = arraysByMatch[i].Split(")),");
                for (int j = 0; j < results.Length; j++)
                {
                    List<string> list = results[j].Split("Array(").ToList<string>();
                    list.RemoveAt(2);
                    list.RemoveAt(0);
                    string[] arrays = list.ToArray();
                    double[][] arrayNumbers = new double[arrays.Length][];
                    //Read the value arrays. The resulting array is in the form [[inputs], [valuesteps1], [valuesteps2]... [timesteps]]
                    for (int k = 0; k < arrays.Length; k++)
                    {
                        arrays[k] = arrays[k].Replace("),", "").Replace(")))", "");
                        string[] numberStrings = arrays[k].Split(",");
                        arrayNumbers[k] = new double[numberStrings.Length];
                        for (int l = 0; l < numberStrings.Length; l++)
                        {
                            arrayNumbers[k][l] = double.Parse(numberStrings[l]);
                        }
                    }
                    //Get values from each value array.
                    //The exported variables from OW include each value at each timestep, but each one has already had a discount rate applied
                    //This algorithm undoes that lowering via overwatchTimeModifier, then allows reapplying time modifiers via newTimeModifier
                    //In effect, this allows the value to be re-calculated for different discount rates
                    double[] finalValues = new double[arrays.Length - 2];
                    for (int k = 1; k < arrayNumbers.Length - 1; k++)
                    {
                        double[] valueSteps = new double[arrayNumbers[k].Length];
                        for (int l = 0; l < valueSteps.Length; l++)
                        {
                            valueSteps[l] = arrayNumbers[k][l] - (l > 0 ? arrayNumbers[k][l] : 0);
                        }

                        double value = 0;

                        for (int l = 0; l < valueSteps.Length; l++)
                        {
                            double originalDiscountRate;
                            double newDiscountRate;
                            if (arrayNumbers[0][49] > 0 || arrayNumbers[0][49] > 0 || arrayNumbers[0][49] > 0)
                            {
                                originalDiscountRate = 0.85;
                                newDiscountRate = 0.85;
                            }
                            else
                            {
                                originalDiscountRate = 0.5;
                                newDiscountRate = 0.5;
                            }
                            value += (valueSteps[l] / Math.Pow(originalDiscountRate, arrayNumbers[arrayNumbers.Length - 1][l])) * Math.Pow(newDiscountRate, arrayNumbers[arrayNumbers.Length - 1][l]);
                        }

                        finalValues[k - 1] = value;
                    }
                    inputDatas.Add(new InputData(arrayNumbers[0], finalValues));
                }
            }
            return inputDatas.ToArray();
        }

        //Converts a set of text files, each containing a set of exported variables from a match, into a single text file that can be used for the other functions.
        static void AssembleOverwatchResults()
        {
            Console.WriteLine("Enter directory to assemble results from:");
            string directory = Console.ReadLine();
            List<string> fileList = Directory.GetFiles(directory, "*.txt").ToList<string>();
            fileList.Remove(directory + "\\results.txt");
            string[] files = fileList.ToArray();
            StreamWriter resultWriter = new(directory + "/results.txt");
            foreach (string file in files)
            {
                bool isFirstResultSetFromFile = true;
                StreamReader sr = new(file);
                string text = sr.ReadToEnd();
                string[] codeLines = text.Split(';');
                string textToAdd = "";
                foreach (string line in codeLines)
                {
                    if (line.Contains("Global.inputResults ="))
                    {
                        if (isFirstResultSetFromFile)
                        {
                            textToAdd += line;
                            isFirstResultSetFromFile = false;
                        } 
                        else
                        {
                            textToAdd = textToAdd.Replace(")))", ")), ");
                            textToAdd += line.Replace("\n\tGlobal.inputResults = Array(", "");
                        }
                    }
                }
                //textToAdd = textToAdd.Substring(2); 
                textToAdd += ";";
                resultWriter.Write(textToAdd);
            }
            resultWriter.Close();
        }

        static void ExportAsOverwatchArray(double[,] w1, double[] w2, double[] wideW)
        {
            const int dp = 7;
            StreamWriter sw = new("export.txt");
            sw.WriteLine("actions");
            sw.WriteLine("{");
            string weights = "\tGlobal.w1Import1 = Array(";
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

            weights = "\tGlobal.w1Import2 = Array(";
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

            weights = "\tGlobal.w2Import = Array(";
            for (int i = 0; i < w2.Length; i++)
            {
                weights += i != w2.Length - 1 ? Math.Round(w2[i], dp) + ", " : Math.Round(w2[i], dp) + ");";
            }
            sw.WriteLine(weights);

            weights = "\tGlobal.wideWeightImport = Array(";
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
            NeuralNetwork NN = new NeuralNetwork(new HiddenLayer[] { new HiddenLayer(1000, 800, 0.1, 0.1) }, new OutputLayer(800, 1, 0.01, 0.1));
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
                NeuralNetwork NN = new NeuralNetwork(new HiddenLayer[] { new HiddenLayer(data[0].inputs.Length, data[0].inputs.Length, 0.1, 0.005) }, new OutputLayer(data[0].inputs.Length, data[0].values.Length, 0.01, 0.001));
                for (int j = 1; j <= 1; j++)
                {
                    foreach (InputData x in data)
                    {
                        NN.backPropagate(x.inputs, x.values, 1 / ((j)));
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
                        loss2 += Math.Abs(x.values[0] - NN.feedForward(x.inputs)[0]);
                        lossCount2++;
                    }
                    else
                    {
                        loss += Math.Abs(x.values[0] - NN.feedForward(x.inputs)[0]);
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
                WideDeepNeuralNetwork WDNN = new WideDeepNeuralNetwork(new HiddenLayer[] { new HiddenLayer(data[0].inputs.Length, data[0].inputs.Length, 0.1, 0.005) }, new WideDeepOutputLayer(data[0].inputs.Length, wideData[0].inputs.Length, data[0].values.Length, 0.01, 0.000001, 0.001));
                for (int j = 1; j <= 3; j++)
                {
                    for (int k = 0; k < data.Length; k++)
                    {
                        WDNN.backPropagate(data[k].inputs, wideData[k].inputs, data[k].values , 1 / ((j)));
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
                        loss2 += Math.Abs(data[k].values[0] - WDNN.wideDeepFeedForward(data[k].inputs, wideData[k].inputs)[0]);
                        lossCount2++;
                    }
                    else
                    {
                        loss += Math.Abs(data[k].values[0] - WDNN.wideDeepFeedForward(data[k].inputs, wideData[k].inputs)[0]);
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
            WideDeepNeuralNetwork WDNN = new WideDeepNeuralNetwork(new HiddenLayer[] { new HiddenLayer(55, 31, 0.01, 0.5) }, new WideDeepOutputLayer(31, 55, 1, 0.01, 0.01, 0.1));
            for (int i = 1; i <= 5; i++)
            {
                for (int j = 0; j < data.Length; j++)
                {
                    WDNN.backPropagate(data[j].inputs, wideData[j].inputs, new double[] { data[j].values[0] }, 1 / ((i)));
                }
            }
            double[] w1 = new double[WDNN.outputLayer.weights.GetLength(0)];
            for (int i = 0; i < WDNN.outputLayer.weights.GetLength(0); i++)
            {
                w1[i] = WDNN.outputLayer.weights[i, 0];
            }
            double[] ww = new double[wideData[0].inputs.Length];
            for (int i = 0; i < ww.Length; i++)
            {
                ww[i] = ((WideDeepOutputLayer)WDNN.outputLayer).wideWeights[i, 0];
            }

            ExportAsOverwatchArray(WDNN.hiddenLayers[0].weights, w1, ww);

            double loss = 0;
            int lossCount = 0;
            double loss2 = 0;
            int lossCount2 = 0;
            for (int k = 0; k < data.Length; k++)
            {
                if (data[k].inputs[49] >= 0.01 || data[k].inputs[50] >= 0.01 || data[k].inputs[51] >= 0.01)
                {
                    loss2 += Math.Abs(data[k].values[0] - WDNN.wideDeepFeedForward(data[k].inputs, wideData[k].inputs)[0]);
                    lossCount2++;
                }
                else
                {
                    loss += Math.Abs(data[k].values[0] - WDNN.wideDeepFeedForward(data[k].inputs, wideData[k].inputs)[0]);
                    lossCount++;
                }
            }
            Console.WriteLine(loss / lossCount);
            Console.WriteLine(loss2 / lossCount2);
            Console.WriteLine((loss + loss2) / (lossCount + lossCount2));
        }

        static void overwatchParityTest()
        {
            HiddenLayer hl = new HiddenLayer(new double[,] { { 0.2, 0.3, 0.1 }, { -0.4, -0.2, -0.1 }, { 0.3, -0.4, -0.2 }, { -0.1, 0.1, 0.3 }, { 0.7, 0.8, 0.6 } }, 0.01);
            WideDeepOutputLayer ol = new WideDeepOutputLayer(new double[,] { { 3 }, { 4 }, {-5 }  }, new double[,] { { -3.2 }, { 2 }, { 4 }, { -1 }, { 2 } }, 0.01, 0.01);
            WideDeepNeuralNetwork wdnn = new WideDeepNeuralNetwork(new HiddenLayer[] { hl }, ol);
            double[] inputs = new double[] { -2, 1, 0.25, 1, 2 };
            Console.WriteLine(wdnn.wideDeepFeedForward(inputs, inputs)[0]);
            wdnn.backPropagate(inputs, inputs, new double[] { -5 }, 1);
            Console.WriteLine(wdnn.wideDeepFeedForward(inputs, inputs)[0]);
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

        static void runPerformanceTests()
        {
            Stopwatch stopwatch = new Stopwatch();
            int largeSize = 1000;
            int smallSize = 200;
            int extraInputSize = 20;
            NeuralNetwork[] networks = new NeuralNetwork[]
            {
                new(new HiddenLayer[] { new HiddenLayer(smallSize, smallSize, 0.1, 0.01) }, new OutputLayer(smallSize, 1, 0.01, 0.01)),
                new(new HiddenLayer[] { new HiddenLayer(smallSize, smallSize, 0.1, 0.01), new HiddenLayer(smallSize, smallSize, 0.1, 0.01) }, new OutputLayer(smallSize, 1, 0.01, 0.01)),
                new(new HiddenLayer[] { new HiddenLayer(largeSize, largeSize, 0.1, 0.01) }, new OutputLayer(largeSize, 1, 0.01, 0.01)),
                new(new HiddenLayer[] { new HiddenLayer(largeSize, largeSize, 0.1, 0.01), new HiddenLayer(largeSize, largeSize, 0.1, 0.01) }, new OutputLayer(largeSize, 1, 0.01, 0.01))
            };

            double[] inputs = new double[largeSize - extraInputSize];
            Random random = new Random();
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = random.NextDouble() - 0.5;
            }
            double[] smallInputs = inputs[0..(smallSize - extraInputSize)];
            double[] extraInputs = inputs[0..extraInputSize];
            double[,] extraInputsArray = new double[25, extraInputSize];
            for (int i = 0; i < extraInputsArray.GetLength(0); i++)
            {
                for (int j = 0; j < extraInputs.Length; j++)
                {
                    extraInputsArray[i, j] = extraInputs[j];
                }
            }

            for (int i = 0; i < networks.Length; i++)
            {
                double[] finalInputs;
                if (networks[i].hiddenLayers[0].weights.GetLength(0) > smallSize) { finalInputs = inputs; }
                else { finalInputs = smallInputs; }
                double time = 0;
                for (int j = 0;j < 10; j++)
                {
                    stopwatch.Reset();
                    stopwatch.Start();
                    networks[i].optimisedFeedForward(finalInputs, extraInputsArray);
                    stopwatch.Stop();
                    time += stopwatch.ElapsedTicks;
                }
                Console.WriteLine(time / 10);

                time = 0;
                stopwatch.Reset();

                List<double> concatInputs = new List<double>();
                concatInputs.AddRange(finalInputs);
                concatInputs.AddRange(extraInputs);
                finalInputs = concatInputs.ToArray();

                for (int j = 0; j < 10; j++)
                {
                    stopwatch.Reset();
                    stopwatch.Start();
                    for (int k = 0; k < extraInputsArray.GetLength(0); k++)
                    {
                        networks[i].feedForward(finalInputs);
                    }
                    stopwatch.Stop();
                    time += stopwatch.ElapsedTicks;
                }
                Console.WriteLine(time / 10);
            }
        }

        static void Main(string[] args)
        {
            runPerformanceTests();
            //AssembleOverwatchResults();
            Console.WriteLine("Enter filepath:");
            string path = Console.ReadLine();
            ReadAdvancedOverwatchResults(path);
            InputData[] data = ReadAdvancedOverwatchResults(path);
            for (int i = 0; i < data.Length; i++)
            {
                //data[i].inputs = data[i].inputs[0..55];
            }
            
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
            testLoss(data, wideData);
            //createOWArray(data, wideData);
            //overwatchParityTest();

        }
    }
}