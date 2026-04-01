DevOps 이해 및 활용 2일차 실습 과제
Github Action 을 활용한 CI/CD Pipeline 구성
(github 내 소스코드(anything) --> 코드 빌드 --> 이미지 빌드 --> Harbor Registry에 이미지 Push
 --> EKS 접속 --> deployment.yaml을 통해 EKS 에 리소스 생성)

-아래의 기본 템플릿을 기반으로 구성할 것. 
-이것은 기본 '템플릿' 이므로 우리 환경 및 설정에 맞게 값들을 바꿔 코드를 재구성해야함.
-Class-2 네임스페이스를 사용해야 함.
name: Build and Deploy to EKS via Harbor

on:
  push:
    branches:
      - main

env:
  APP_NAME: sample-app

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    env:
      HARBOR_REGISTRY: ${{ secrets.HARBOR_REGISTRY }}
      HARBOR_PROJECT: ${{ secrets.HARBOR_PROJECT }}
      HARBOR_USERNAME: ${{ secrets.HARBOR_USERNAME }}
      HARBOR_PASSWORD: ${{ secrets.HARBOR_PASSWORD }}

      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      AWS_REGION: ${{ secrets.AWS_REGION }}

      EKS_CLUSTER_NAME: ${{ secrets.EKS_CLUSTER_NAME }}
      K8S_NAMESPACE: ${{ secrets.K8S_NAMESPACE }}

    steps:
      - name: Checkout source
        uses: actions/checkout@v4

      - name: Set image metadata
        run: |
          SHORT_SHA=$(echo $GITHUB_SHA | cut -c1-7)
          IMAGE_TAG=${GITHUB_RUN_NUMBER}-${SHORT_SHA}
          IMAGE_REPO=${HARBOR_REGISTRY}/${HARBOR_PROJECT}/${APP_NAME}
          FULL_IMAGE=${IMAGE_REPO}:${IMAGE_TAG}

          echo "IMAGE_TAG=${IMAGE_TAG}" >> $GITHUB_ENV
          echo "IMAGE_REPO=${IMAGE_REPO}" >> $GITHUB_ENV
          echo "FULL_IMAGE=${FULL_IMAGE}" >> $GITHUB_ENV

      - name: Build application
        run: |
          echo "Replace with your actual build command"

          # Gradle example
          # chmod +x gradlew
          # ./gradlew clean build -x test

          # Maven example
          # mvn clean package -DskipTests

          # Node.js example
          # npm ci
          # npm run build

      - name: Log in to Harbor
        run: |
          echo "${HARBOR_PASSWORD}" | docker login ${HARBOR_REGISTRY} -u "${HARBOR_USERNAME}" --password-stdin

      - name: Build Docker image
        run: |
          docker build -t ${FULL_IMAGE} .

      - name: Push Docker image to Harbor
        run: |
          docker push ${FULL_IMAGE}

      - name: Install kubectl
        uses: azure/setup-kubectl@v4
        with:
          version: 'latest'

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Update kubeconfig for EKS
        run: |
          aws eks update-kubeconfig \
            --region ${AWS_REGION} \
            --name ${EKS_CLUSTER_NAME}

      - name: Render deployment manifest
        run: |
          cp deploy/deployment.yaml deploy/deployment-rendered.yaml
          sed -i "s|__IMAGE__|${FULL_IMAGE}|g" deploy/deployment-rendered.yaml

      - name: Deploy to EKS
        run: |
          kubectl apply -n ${K8S_NAMESPACE} -f deploy/deployment-rendered.yaml
          kubectl apply -n ${K8S_NAMESPACE} -f deploy/service.yaml

      - name: Check rollout status
        run: |
          kubectl rollout status deployment/${APP_NAME} -n ${K8S_NAMESPACE} --timeout=300s
          kubectl get pods -n ${K8S_NAMESPACE}
          kubectl get svc -n ${K8S_NAMESPACE}

      - name: Logout Harbor
        if: always()
        run: |
          docker logout ${HARBOR_REGISTRY} || true