pipeline {
  agent any

  stages {
    stage('build-backend') {
      steps {
        sh 'docker build -t my-backend ./backend'
      }
    }

    stage('build-frontend') {
      steps {
        sh 'docker build -t my-frontend ./frontend'
      }
    }

    stage('tag') {
      steps {
        sh '''
        docker tag my-backend amdp-registry.skala-ai.com/skala26a-ai2/my-backend:latest
        docker tag my-frontend amdp-registry.skala-ai.com/skala26a-ai2/my-frontend:latest
        '''
      }
    }

    stage('push') {
      steps {
        sh '''
        docker login amdp-registry.skala-ai.com \
        -u 'robot$skala26a-ai2' \
        -p Va9M8WvbaoPa4oxpqFHMH4TH0h02GbTH

        docker push amdp-registry.skala-ai.com/skala26a-ai2/my-backend:latest
        docker push amdp-registry.skala-ai.com/skala26a-ai2/my-frontend:latest
        '''
      }
    }
  }
}