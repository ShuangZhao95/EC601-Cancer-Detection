(function() {
	
	

  	const txtEmail = document.getElementById('txtEmail');
  	const txtPassword = document.getElementById('txtPassword');
  	const btnLogin = document.getElementById('btnLogin');
  	const btnSignUp = document.getElementById('btnSignUp');
  	const btnLogout = document.getElementById('btnLogout');
    const uploader = document.getElementById('uploader');
    const fileButton = document.getElementById('fileButton');

  	btnLogin.addEventListener('click', e=>{
  		const email = txtEmail.value;
  		const pass = txtPassword.value;
  		const auth = firebase.auth();

  		const promise = auth.signInWithEmailAndPassword(email, pass);
  		promise.catch(e=>console.log(e.message));

  	});

  	btnSignUp.addEventListener('click', e=>{
  		//TODO:CHECK 4 REAL EMAIL
  		const email = txtEmail.value;
  		const pass = txtPassword.value;
  		const auth = firebase.auth();
  		const promise = auth.createUserWithEmailAndPassword(email, pass);
  		promise.catch(e=>console.log(e.message));

  	});

  	btnLogout.addEventListener('click', e=>{
  		firebase.auth().signOut();
  	});

  	firebase.auth().onAuthStateChanged(firebaseUser=>{
  		if(firebaseUser) {
  			console.log(firebaseUser);
  			btnLogout.classList.remove('hide');
        btnSignUp.classList.add('hide');
        btnLogin.classList.add('hide');
        fileButton.classList.remove('hide');
        uploader.classList.remove('hide');
        uploadmsg.classList.remove('hide');
        loginmsg.classList.add('hide');
  		} else {
  			console.log('not logged in');
  			btnLogout.classList.add('hide');
        btnSignUp.classList.remove('hide');
        btnLogin.classList.remove('hide');
        fileButton.classList.add('hide');
        uploader.classList.add('hide');
        loginmsg.classList.remove('hide');
        uploadmsg.classList.add('hide');
  		}
  	});






      fileButton.addEventListener('change', function(e) {

        var file = e.target.files[0];

        var storageRef = firebase.storage().ref('textfortest/' + file.name);

        var task = storageRef.put(file);
        task.on('state_changed',

          function progress(snapshot) {
            var percentage = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
            uploader.value = percentage;
          },

          function error(err) {

          },

          function complete() {

          }
        );

      });

}());