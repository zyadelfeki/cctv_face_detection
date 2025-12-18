#!/usr/bin/env bash
# Kubernetes Deployment Script
# Usage: ./scripts/deploy-k8s.sh [environment]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
NAMESPACE="cctv-detection"
KUBECONFIG=${KUBECONFIG:-~/.kube/config}

echo -e "${GREEN}Starting Kubernetes deployment for environment: ${ENVIRONMENT}${NC}"

# Validate prerequisites
command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}kubectl is required but not installed. Aborting.${NC}" >&2; exit 1; }
command -v helm >/dev/null 2>&1 || { echo -e "${YELLOW}Warning: helm not found. Some features may not work.${NC}"; }

# Check cluster connectivity
echo "Checking cluster connectivity..."
if ! kubectl cluster-info &>/dev/null; then
    echo -e "${RED}Cannot connect to Kubernetes cluster. Check your kubeconfig.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Connected to cluster${NC}"

# Create namespace if it doesn't exist
echo "Creating namespace ${NAMESPACE}..."
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Apply namespace configuration
echo "Applying namespace labels and policies..."
kubectl apply -f k8s/namespace.yaml

# Apply resource quotas and limits
echo "Applying resource quotas..."
kubectl apply -f k8s/resource-quota.yaml

# Apply network policies
echo "Applying network policies (zero-trust)..."
kubectl apply -f k8s/network-policy.yaml

# Check if secrets exist
echo "Checking secrets..."
if ! kubectl get secret cctv-secrets -n ${NAMESPACE} &>/dev/null; then
    echo -e "${YELLOW}Warning: cctv-secrets not found. Please create secrets before deploying.${NC}"
    echo "Example: kubectl create secret generic cctv-secrets --from-literal=database-url=... -n ${NAMESPACE}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Deploy PostgreSQL
echo "Deploying PostgreSQL..."
kubectl apply -f k8s/postgres-statefulset.yaml

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=300s

# Deploy Redis
echo "Deploying Redis..."
kubectl apply -f k8s/redis-deployment.yaml

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
kubectl wait --for=condition=ready pod -l app=redis -n ${NAMESPACE} --timeout=300s

# Deploy API
echo "Deploying API server..."
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml

# Wait for API to be ready
echo "Waiting for API to be ready..."
kubectl wait --for=condition=ready pod -l app=cctv-api -n ${NAMESPACE} --timeout=300s

# Deploy Ingress
if [ "${ENVIRONMENT}" = "production" ]; then
    echo "Deploying ingress..."
    kubectl apply -f k8s/ingress.yaml
fi

# Display deployment status
echo -e "\n${GREEN}Deployment complete!${NC}"
echo -e "\nPod status:"
kubectl get pods -n ${NAMESPACE}

echo -e "\nService status:"
kubectl get svc -n ${NAMESPACE}

if [ "${ENVIRONMENT}" = "production" ]; then
    echo -e "\nIngress status:"
    kubectl get ingress -n ${NAMESPACE}
fi

echo -e "\n${GREEN}✓ Deployment successful!${NC}"
echo -e "\nTo view logs: kubectl logs -f -l app=cctv-api -n ${NAMESPACE}"
echo -e "To get a shell: kubectl exec -it deployment/cctv-api -n ${NAMESPACE} -- sh"
