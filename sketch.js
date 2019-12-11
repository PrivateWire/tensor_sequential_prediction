function LayersAPI() {

    //This the model
    const model = tf.sequential();

    // Create hidden layer
    //Dense is a fully connected layer
    const hidden = tf.layers.dense({
       units:4, //number of nodes
       inputShape:[2], // input shape
       activation:'sigmoid'//activation function
   });

    // Add layer
    model.add(hidden);

    //Create another output layer - input shape is inferred from input
    const output = tf.layers.dense({
       units:1, //Number of outputs to the console.
       activation:'sigmoid',
   });

    //Add it
    model.add(output);

    //An optimiser is used to add gradient descent
    const sgdOpt = tf.train.sgd(0.5);

    //Compile
    model.compile({
        optimizer:sgdOpt,
        loss:tf.losses.meanSquaredError,
    });


    //Training data -inputs could be from spreadhseet etc

    const xs = tf.tensor2d([[0,0],[0.5,0.5],[1,1]]);

    //Known outputs
    const ys = tf.tensor2d([[1],[0.5],[0]]);


    //Run training
    train().then(()=>{
        console.log('training complete');

        //Make prediction
        let outputs = model.predict(xs);
        outputs.print();

    })

    //async
    async function train() {

        for(let i = 0; i < 1000; i++){

            const config={
                shuffle:true,
                epoch:100
            }
            const response = await model.fit(xs,ys,config);
            console.log(response.history.loss[0]);
        }

    }



}

console.log(LayersAPI());
