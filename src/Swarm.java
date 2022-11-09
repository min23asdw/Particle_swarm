import java.util.ArrayList;
import java.util.Random;
public class Swarm {

    String[] mlp;
    String neural_type;
    double biases;
    double learningRate = 0.1;
    int num_population;
    int maxEpoch;

    double pbest = 99999999;
    Matrix[] weight_pbest;
//    Matrix[] vector_weight;
    Random r = new Random();
    individual[] seach_space;

    individual[] selection_pool ;

    ArrayList<Double[]> train_dataset;
    ArrayList<Double[]> train_desired_data;

    ArrayList<Double[]> train_result = new ArrayList<>();

    ArrayList< Pair<Double , individual>> scoreBoard = new ArrayList<>();

//    double prob_mul = 0.01;
//    double prob_parent = 0.3;
//    double best_mlp_score = 0.0;


    public Swarm(String _mlp , double _biases , int _population , int _maxGeneration){
        this.neural_type = _mlp;
        this.mlp = _mlp.split(",");;
        this.biases = _biases;
        this.num_population = _population;
        this.maxEpoch = _maxGeneration;

    }

    public void settraindata(ArrayList<Double[]> _train_dataset, ArrayList<Double[]> _train_desired_data) {
        this.train_dataset = _train_dataset;
        this.train_desired_data = _train_desired_data;
    }

    public void init_Particle(){
        individual[] init =  new individual[num_population];
        for(int indi = 0 ; indi < num_population ; indi++){
            individual indiler = new individual( neural_type ,biases );
            init[indi] = indiler;
        }
        this.seach_space = init;
        this.selection_pool = new individual[num_population];

        this.weight_pbest = newWeight(seach_space[0]);
//        this.vector_weight =  newWeight(seach_space[0]);
    }

    public void run_gen(){
//        System.out.println("train");
        init_Particle();

        for (int epoch = 0; epoch < maxEpoch; epoch++){

            Double[] result =  paticle_eval(train_dataset,train_desired_data);
            System.out.println(result[0] + "\t" + result[1] + "\t" + result[2] );
            train_result.add(result);


            //step 4
            for (individual particle : seach_space) {
                Matrix[] particle_weight = particle.get_weight();

                Matrix[] vector_weight = newWeight(particle);
                for (int layer = 0 ; layer < particle.layer_weight.length ; layer++) {
                    Matrix a = Matrix.sub_matrix(weight_pbest[layer],particle_weight[layer]);
                    Matrix b = Matrix.mul_matrix(a,learningRate);
                    Matrix c = Matrix.plus_matrix(b,particle.vector_weight[layer]);
                    vector_weight[layer] = c;
//                    System.out.println("");
                }
                particle.set_vector_weight(vector_weight);
            }

            //step 5
            for (individual particle : seach_space) {

                Matrix[] particle_weight = particle.get_weight();
                Matrix[] vector_weight = particle.get_vector_weight();

                Matrix[] new_particle_weight = newWeight(particle);
                for (int layer = 0 ; layer < particle.layer_weight.length ; layer++) {
                    Matrix a = Matrix.plus_matrix(particle_weight[layer], vector_weight[layer]);
                    new_particle_weight[layer] = a;
                }
                particle.set_weight(new_particle_weight);
            }



//            vector_weight =  vector_weight +  learningRate * (weight_pbest - getWeight)

//            selection();
//            ArrayList<individual> offspring_pool =  p1();
//            p2(offspring_pool);
//            add2N(offspring_pool);

//            if(gen != maxEpoch -1)  move2next_population(offspring_pool);

        }
        System.out.println("");

    }

    public void test(ArrayList<Double[]> dataset, ArrayList<Double[]> desired_data){
//        System.out.println("test");
        individual best_solution =  scoreBoard.get(scoreBoard.size()-1).particle;
        best_solution.test(dataset,desired_data);
//        System.out.println("test");

    }

    public Double[] paticle_eval(ArrayList<Double[]> dataset , ArrayList<Double[]> desired_data){
        double sum_fit = 0;
        double avg_fit;
        double min_fit = 9999999;

        //eval fitness
        for (individual particle : seach_space) {
            double error_mlp =   particle.eval(dataset , desired_data);

            double fitness = error_mlp;
            if(min_fit > fitness){
                min_fit = fitness;
            }

            sum_fit += fitness;
            //step 3
            if(fitness < pbest){
                pbest = fitness;
                //TODO  clone?
                weight_pbest = particle.get_weight();
                Pair<Double , individual> score_particle = new Pair<>(fitness, particle);
                scoreBoard.add(score_particle);
            }

        }
        avg_fit = sum_fit / num_population;
        return new Double[]{sum_fit, avg_fit, min_fit};
    }

    public double scaling(double error){
        return 1/(error + 0.01)  ;
    }

//    public void selection(){
//        // random tournament selection
//
//        for(int i = 0 ; i < num_population ; i++) {
//
//            int select1 = r.nextInt(0,49);
//            int select2 = r.nextInt(0,49);
//            individual a1 = seach_space[select1].clone();
//            individual a2 = seach_space[select2].clone();
//
//            double a1_fit = scaling(a1.avg_error_n);
//            double a2_fit = scaling(a2.avg_error_n);
//
//            if(a1_fit > a2_fit  ) selection_pool[i] = a1;
//            else  selection_pool[i] = a2;
//        }
//    }

//    public  ArrayList<individual>  p1(){
//        ArrayList<individual> offspring_pool = new ArrayList<>();
//        //pair selection
//        for(int ran = 0 ; ran < num_population/2 ; ran++){
//            int select1 = r.nextInt(0,selection_pool.length);
//            int select2 = r.nextInt(0,selection_pool.length);
//
//            individual offspring =  crossover(selection_pool[select1], selection_pool[select2]);
//
//            offspring_pool.add(offspring);
//        }
//        return offspring_pool;
//    }
//
//    public individual crossover(individual f , individual m){
//        Matrix[] father_weight = f.clone().get_weight();
//        Matrix[] mother_weight = m.clone().get_weight();
//
//        individual offspring = new individual( neural_type ,biases );
//
//        Matrix[] offspring_weight = newWeight(f);
//        for (int layer = 0 ; layer < father_weight.length ; layer++ ) {
//            for (int node = 0; node < father_weight[layer].rows; node++) {
//                double q = uniform_random(0.0,1.0);
//                if (q < prob_parent) {  // เอาจากพ่อ
//                    offspring_weight[layer].data[node] = father_weight[layer].data[node].clone();
//                } else {  //เอาจากแม่
//                    offspring_weight[layer].data[node] = mother_weight[layer].data[node].clone();
//                }
//            }
//        }
//        offspring.set_weight(offspring_weight);
//
//        return offspring;
//    }

    public Matrix[] newWeight(individual blueprint){
        Matrix[] offsprong_weight = new Matrix[blueprint.neural_type.length-1];
        for (int layer = 0; layer < offsprong_weight.length; layer++) {
            Matrix weight = new Matrix(blueprint.neural_type[layer+1],blueprint.neural_type[layer] ,false);
            offsprong_weight[layer] = weight;
        }
        return offsprong_weight;
    }

//    public void p2(ArrayList<individual> offspring_pool){
//        for (individual offspring:offspring_pool) {
//            Matrix[] nodeofchild = offspring.get_weight();
//            for (int layer = 0 ; layer < nodeofchild.length ; layer++ ){
//                for (int node = 0 ; node < nodeofchild[layer].rows ; node++) {
//                    //            for each non-input node
//                    double q = uniform_random(0.0,1.0);
//
//                    if( q < prob_mul){
//                        mutation(offspring,layer,node);
//                    }
//                }
//            }
//        }
//    }

//    public void mutation(individual offspring , int layer , int node){
//        Matrix[] a = offspring.get_weight();
//        for (int weightline = 0 ; weightline < a[layer].cols ; weightline ++ ){
//            double e = uniform_random(-1.0,1.0);
//            offspring.add_weight(layer,node,weightline , e);
//        }
//    }

//    public void add2N(ArrayList<individual> offspring_pool ){
//        while(offspring_pool.size() < num_population){
//            int pick = r.nextInt(0,num_population-1);
//            individual copy = seach_space[pick].clone();
//
//            offspring_pool.add(copy);
//        }
//    }

//    public void move2next_population(ArrayList<individual> offspring_pool) {
//        int i = 0 ;
//        for (individual offspring : offspring_pool) {
//            seach_space[i] =  offspring.clone();
//            i++;
//        }
//    }
//
//    public double uniform_random(double rangeMin , double rangeMax ){
//        return rangeMin + (rangeMax - rangeMin) * r.nextDouble();
//    }

}
