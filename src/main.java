import java.io.*;
import java.util.ArrayList;

public class main {

    private static ArrayList<ArrayList<Double[]>> test_dataset = new ArrayList<>();
    private static ArrayList<ArrayList<Double[]>> test_desired_data = new ArrayList<>();

    private static ArrayList<ArrayList<Double[]>> train_dataset = new ArrayList<>();
    private static ArrayList<ArrayList<Double[]>> train_desired_data = new ArrayList<>();

    private static int NumberOftest = 1;

    public static void main(String[] args) throws IOException {
        for(int tain_i = 0 ; tain_i < NumberOftest ; tain_i ++) {
            ArrayList<Double[]> test_dataset_i = new ArrayList<>();
            ArrayList<Double[]> test_desired_data_i = new ArrayList<>();

            ArrayList<Double[]> train_dataset_i = new ArrayList<>();
            ArrayList<Double[]> train_desired_data_i = new ArrayList<>();

            FileInputStream fstream = new FileInputStream("src/AirQualityUCI_dataset.txt");
            DataInputStream in = new DataInputStream(fstream);
            BufferedReader br = new BufferedReader(new InputStreamReader(in));
            String data;

            int line_i = 1;
            while ((data = br.readLine()) != null) { // each line

                String[] eachLine = data.split("\t");

                Double[] temp_input = new Double[8];
                Double[] ans = new Double[1];

//                for(int  i = 2 ; i<eachLine.length;i++){
//                    temp_input[i-2] = Double.parseDouble(eachLine[i])/6000.8;  // lazy min max norm
//                }
                temp_input[0] = (Double.parseDouble(eachLine[3])+200)/(2050.0+200);
                temp_input[1] = (Double.parseDouble(eachLine[6])+200)/(2300.0+200);
                temp_input[2] = (Double.parseDouble(eachLine[8])+200)/(2700.0+200);
                temp_input[3] = (Double.parseDouble(eachLine[10])+200)/(2800.0+200);
                temp_input[4] = (Double.parseDouble(eachLine[11])+200)/(2600.0+200);
                temp_input[5] = (Double.parseDouble(eachLine[12])+200)/(50.0+200);
                temp_input[6] = (Double.parseDouble(eachLine[13])+200)/(90.0+200);
                temp_input[7] = (Double.parseDouble(eachLine[14])+200)/(3.0+200);

                ans[0] =  (Double.parseDouble(eachLine[5])+200)/(70.0+200);

//                if(eachLine[1].equals("B")){ ans = new Double[]{0.0, 1.0};   // b 01
//                }
//                else if (eachLine[1].equals("M")){ans = new Double[]{1.0, 0.0};  // m 10
//                }

                if (line_i % 10 == tain_i) {  // 10% for test
                    test_dataset_i.add(temp_input);
                    test_desired_data_i.add(ans);
                } else {
                    train_dataset_i.add(temp_input);
                    train_desired_data_i.add(ans);
                }

                line_i++;
            }
            test_desired_data.add(test_desired_data_i);
            train_desired_data.add(train_desired_data_i);
            test_dataset.add(test_dataset_i);
            train_dataset.add(train_dataset_i);
        }
        System.out.println("work");

        for(int test_i = 0 ; test_i < NumberOftest ; test_i ++) {
//            System.out.println("===================================================");
            Swarm ga = new Swarm("8,5,1",  0, 50 , 1000 );
            System.out.println("train: " + test_i);
            ga.settraindata( train_dataset.get(test_i), train_desired_data.get(test_i));
            ga.run_gen();
            System.out.println("test: " + test_i);
            ga.test(test_dataset.get(test_i), test_desired_data.get(test_i));
//            System.out.println("===================================================");
        }
    }
}
