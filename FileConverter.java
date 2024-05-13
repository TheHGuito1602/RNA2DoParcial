import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Arrays;

public class FileConverter {
    private static final ArrayList<float[]> Weight = new ArrayList<>();// Lista de Arreglos de pesos
    private static final ArrayList<float[]> Delta = new ArrayList<>();// Lista de Arreglos de deltas
    private static final ArrayList<int[]> F_Net = new ArrayList<>();// Lista de Arreglos de FNet
    private static final ArrayList<int[]> Y = new ArrayList<>();// Lista de Arreglos de las Y
    private static final ArrayList<float[]> A_Net = new ArrayList<>();//

    public static void main(String[] args) {
        // Declaración de variables
        List<int[]> datos = readFileAndConvert("TRAIN.txt", 63);
        List<int[]> clase = readFile("ETIQUETAS.txt", 7);
        int epocas = 10;
        float alpha = 9.5f;
        int bias = 1;
        float[] w_inicial;// Variables iniciales y auxiliares
        double per;

        for (int a = 0; a < 7; a++) {
            w_inicial = iniciarRandom();
            Weight.add(w_inicial);
        }

        for (int i = 0; i < epocas; i++) {// Ciclo para recorrer épocas
            System.out.println("Epoca: " + (i + 1));// Impresion de epoca
            for (int j = 0; j < datos.size(); j++) {// Ciclo para recorrer los 21 registros

                int[] registro = datos.get(j);// Registro de 63 entradas
                int[] etiqueta = clase.get(j);// Etiqueta de 7 salidas
                float[] net = new float[7];
                int[] fNet = new int[7], y = new int[7];
                
                for (int k = 0; k < 7; k++) {// Ciclo para recorrer las 7 salidas
                    net[k] = NET(registro, Weight.get(k), bias);// Obtener el Net de una salida y la entrada
                                                                // respectivamente
                    fNet[k] = FuncionNet(net[k]);// Obtener el F_Net de una salida
                    y[k] = fNet[k];// Salida generada es igual al fNet
                }

                A_Net.add(net);// Registro de Activación por Entrada
                F_Net.add(fNet);// Registro de Transferencia por Entrada
                Y.add(y);// Registro de salidas generadas por Entrada
                ArrayList<float[]> WeightC = clonar(Weight);// Lista de Arreglos de pesos
                Weight.clear();
                Delta.clear();
                for (int k = 0; k < 7; k++) {
                    Delta.add(etapaAprendizaje(alpha, registro, bias, etiqueta[k], y[k]));
                    Weight.add(nuevosPesos(WeightC.get(k), Delta.get(k)));
                }
                //System.out.println("Peso: "+Weight.size()+" | Delta: "+Delta.size());
            }
            per = porcentaje(clase, Y);
            if (per >= 100) {
                System.out.println("Epoca: " + (i+1) + " y porcentaje: " + per);
                prueba(alpha,Weight, bias);
                break;
            }
            limpiarEpoca();
        }
    }

    private static void prueba(float alpha, ArrayList<float[]> weight, int bias) {
        List<int[]> datos = readFileAndConvert("TEST.txt", 63);
        List<int[]> clase = readFile("ETIQUETAS.txt", 7);
        double per;
        Y.clear();
        for (int j = 0; j < datos.size(); j++) {// Ciclo para recorrer los 21 registros

            int[] registro = datos.get(j);// Registro de 63 entradas
            int[] etiqueta = clase.get(j);// Etiqueta de 7 salidas
            float[] net = new float[7];
            int[] fNet = new int[7], y = new int[7];
            
            for (int k = 0; k < 7; k++) {// Ciclo para recorrer las 7 salidas
                net[k] = NET(registro, Weight.get(k), bias);// Obtener el Net de una salida y la entrada
                                                            // respectivamente
                fNet[k] = FuncionNet(net[k]);// Obtener el F_Net de una salida
                y[k] = fNet[k];// Salida generada es igual al fNet
            }

            A_Net.add(net);// Registro de Activación por Entrada
            F_Net.add(fNet);// Registro de Transferencia por Entrada
            Y.add(y);// Registro de salidas generadas por Entrada
            ArrayList<float[]> WeightC = clonar(Weight);// Lista de Arreglos de pesos
            Weight.clear();
            Delta.clear();
            for (int k = 0; k < 7; k++) {
                Delta.add(etapaAprendizaje(alpha, registro, bias, etiqueta[k], y[k]));
                Weight.add(nuevosPesos(WeightC.get(k), Delta.get(k)));
            }
            //System.out.println("Peso: "+Weight.size()+" | Delta: "+Delta.size());
        }
        per = porcentaje(clase, Y);
        System.out.println("Porcentaje en TRAIN: "+per);
    }

    private static ArrayList<float[]> clonar(ArrayList<float[]> original) {
        ArrayList<float[]> clone = new ArrayList<>();
        for (float[] array : original) {
            // Clonar cada arreglo de flotantes individualmente
            float[] clonedArray = array.clone();
            clone.add(clonedArray);
        }
        return clone;
    }

    public static List<int[]> readFileAndConvert(String filename, int d) {
        List<int[]> listaLetras = new ArrayList<>();// Lista para guardar los arreglos en letras
        int[] patronLetra = new int[d];// Arreglo para cada letra
        int count = 0;// Contador

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {// Intentar leer el archivo
            String line;// Linea para capturar
            while ((line = br.readLine()) != null) {// Mientras la linea de captura sea diferente de null
                for (int i = 0; i < line.length(); i++) {// Ciclo for de acuerdo a la longitud del contenido
                    char c = line.charAt(i);// Variable para capturar caracter a caracter
                    int value = convertCharToValue(c);// convertir caracter a numero
                    if (count < d) {// Si el contador es menor a la dimension
                        patronLetra[count++] = value;// El patron se llena con el valor
                    }
                    // Si se completa un conjunto de d elementos, agregar una copia a la lista y
                    // reiniciar
                    if (count == d) {
                        listaLetras.add(Arrays.copyOf(patronLetra, patronLetra.length));// Agregar arreglo
                        patronLetra = new int[d];// Genera nuevo arreglo
                        count = 0;// Reinicia el contador
                    }
                }
            }
            // Si queda un conjunto de menos de 64 elementos al final del archivo, agregarlo
            // también
            if (count > 0) {
                listaLetras.add(Arrays.copyOf(patronLetra, count)); // Solo copiamos los elementos válidos
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return listaLetras;
    }

    public static List<int[]> readFile(String filePath, int d) {
        List<int[]> resultList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                int[] binaryArray = new int[d];
                for (int i = 0; i < binaryArray.length; i++) {
                    binaryArray[i] = Character.getNumericValue(line.charAt(i));
                }
                resultList.add(binaryArray);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return resultList;
    }

    public static int convertCharToValue(char c) {
        switch (c) {
            case '#':
            case '@':
                return 1;
            case '.':
            case 'o':
                return -1;
            default:
                return 0; // Otra opción por si encuentras un caracter desconocido
        }
    }

    public static void imprimir(List<int[]> lista) {
        for (int i = 0; i < lista.size(); i++) {
            System.out.println(Arrays.toString(lista.get(i)));
        }
    }

    private static float NET(int[] x, float[] w, int b) {// Función de activación
        float net = 0;// Variable auxiliar
        for (int i = 0; i < w.length; i++) {
            if (i >= x.length) {// Condición para determinar si de entrada o Bias
                net = net + (w[i] * b);// Net = Net + (W * x)
            } else {
                net = net + (w[i] * x[i]);
            }
        }
        return net;
    }

    private static int FuncionNet(float net) {// Función de transferencia
        if (net > 0) {// Determina valor de Función Net
            return 1;
        } else {
            return 0;
        }
    }

    private static float[] etapaAprendizaje(float a, int[] x, int b, int d, int y) {// Etapa de aprendizaje
        float[] deltas = new float[x.length + 1];// Deltas de acuerdo a las entradas y el Bias
        for (int i = 0; i < deltas.length; i++) {
            if (i >= x.length) {// Condición para determinar si de entrada o Bias
                deltas[i] = a * b * (d - y);// Delta = Alpha * x * (d - y)
            } else {
                deltas[i] = a * x[i] * (d - y);
            }
        }
        return deltas;
    }

    private static float[] nuevosPesos(float[] pesos, float[] deltas) {// Actualizar pesos
        // float[] w_new = new float[pesos.length];// Variable auxiliar
        for (int i = 0; i < pesos.length; i++) {
            pesos[i] = pesos[i] + deltas[i];// Nuevo W = W anterior + Delta del W anterior
        }
        return pesos;
    }

    public static double porcentaje(List<int[]> c, ArrayList<int[]> Y) {
        int match = 0; // Contador para coincidencias
        int total = c.size(); // Tamaño de la lista c
        double percent; // Porcentaje de coincidencia

        for (int i = 0; i < c.size(); i++) {
            int[] arrayC = c.get(i);
            int[] arrayY = Y.get(i);
            System.out.print("Combinación: " + (i + 1) + "\t|\t");
            System.out.println("Salida esperada ->" + Arrays.toString(arrayC) + " VS " + Arrays.toString(arrayY)
                    + "<- Salida aprendida");
            if (arrayC.length == arrayY.length) {
                boolean isEqual = true;
                for (int j = 0; j < arrayC.length; j++) {
                    if (arrayC[j] != arrayY[j]) { // Comparación de valores de los arreglos
                        isEqual = false;
                        break;
                    }
                }
                if (isEqual) {
                    match++;
                }
            }
        }

        percent = ((double) match / total) * 100;

        // Formatear el porcentaje con dos decimales
        String formattedPercent = String.format("%.2f", percent);
        System.out.println("Caracteres reconocidos: " + match);
        System.out.println("Caracteres no reconocidos: "+(total-match));
        System.out.println("Porcentaje de coincidencia: " + formattedPercent + "%");
        return Double.parseDouble(formattedPercent);
    }

    private static void limpiarEpoca() {// Borrar registros
        // Weight.clear();
        A_Net.clear();
        F_Net.clear();
        Y.clear();
        Delta.clear();
    }

    private static float[] iniciarZeros() {
        float[] w_inicial = new float[64];
        for (int i = 0; i < w_inicial.length; i++) {
            w_inicial[i] = 0.0f;
        }
        return w_inicial;
    }

    private static float[] iniciarRandom() {
        float[] w_inicial = new float[64];
        Random rand = new Random();
        for (int i = 0; i < w_inicial.length; i++) {
            float randomValue = Math.round(rand.nextFloat() * 100) / 100.0f; // Generar valor aleatorio con dos
                                                                             // decimales
            w_inicial[i] = randomValue;
        }
        return w_inicial;
    }

}
