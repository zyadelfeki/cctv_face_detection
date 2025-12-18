# Kubernetes Deployment Guide

## 🚀 Production-Ready CCTV Face Detection System

This directory contains production-hardened Kubernetes manifests for deploying the CCTV Face Detection system with maximum security.

## 📋 Prerequisites

### Required Components

1. **Kubernetes Cluster** (v1.24+)
   - 3+ worker nodes (4 CPU, 16GB RAM minimum each)
   - SSD storage class available
   - LoadBalancer or Ingress controller

2. **Tools**
   ```bash
   kubectl v1.24+
   helm v3.10+
   kustomize v4.5+ (optional)
   ```

3. **Required Kubernetes Add-ons**
   ```bash
   # Ingress Controller (NGINX)
   helm install ingress-nginx ingress-nginx/ingress-nginx \
     --namespace ingress-nginx --create-namespace
   
   # Cert Manager (for TLS)
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
   
   # Metrics Server (for HPA)
   kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
   
   # External Secrets Operator (recommended)
   helm install external-secrets external-secrets/external-secrets \
     --namespace external-secrets-system --create-namespace
   ```

4. **Secrets Management**
   - AWS Secrets Manager, or
   - HashiCorp Vault, or
   - Sealed Secrets

## 🔐 Security Configuration

### 1. Create Namespace with Security

```bash
kubectl apply -f namespace.yaml
```

This creates:
- Isolated namespace with Pod Security Standards (restricted)
- Resource quotas to prevent resource exhaustion
- Limit ranges for default constraints

### 2. Configure Secrets

**Option A: External Secrets Operator (Recommended)**

```bash
# Store secrets in AWS Secrets Manager first
aws secretsmanager create-secret \
  --name cctv/production/database \
  --secret-string '{"username":"cctv_user","password":"STRONG_PASSWORD","database":"cctv_db","url":"postgresql://cctv_user:STRONG_PASSWORD@postgres:5432/cctv_db"}'

aws secretsmanager create-secret \
  --name cctv/production/encryption \
  --secret-string '{"master_key":"'$(openssl rand -base64 32)'"}'

aws secretsmanager create-secret \
  --name cctv/production/auth \
  --secret-string '{"jwt_secret":"'$(openssl rand -base64 64)'","api_key_salt":"'$(openssl rand -base64 32)'"}'

aws secretsmanager create-secret \
  --name cctv/production/redis \
  --secret-string '{"password":"'$(openssl rand -base64 32)'","url":"redis://:PASSWORD@redis:6379/0"}'

# Apply External Secrets
kubectl apply -f secrets.yaml
```

**Option B: Manual Secrets (Development Only)**

```bash
# Database credentials
kubectl create secret generic cctv-db-credentials \
  --from-literal=POSTGRES_USER=cctv_user \
  --from-literal=POSTGRES_PASSWORD=STRONG_PASSWORD \
  --from-literal=POSTGRES_DB=cctv_db \
  --from-literal=DATABASE_URL=postgresql://cctv_user:STRONG_PASSWORD@postgres:5432/cctv_db \
  --namespace=cctv-system

# Encryption keys
kubectl create secret generic cctv-api-keys \
  --from-literal=ENCRYPTION_MASTER_KEY=$(openssl rand -base64 32) \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 64) \
  --from-literal=API_KEY_SALT=$(openssl rand -base64 32) \
  --namespace=cctv-system

# Redis credentials
REDIS_PASSWORD=$(openssl rand -base64 32)
kubectl create secret generic cctv-redis-credentials \
  --from-literal=REDIS_PASSWORD=$REDIS_PASSWORD \
  --from-literal=REDIS_URL="redis://:${REDIS_PASSWORD}@redis:6379/0" \
  --namespace=cctv-system
```

### 3. Configure TLS Certificates

**Option A: Let's Encrypt (Recommended)**

```bash
# Update ingress.yaml with your domain
sed -i 's/cctv.company.com/your-domain.com/g' ingress.yaml
sed -i 's/security@company.com/your-email@company.com/g' ingress.yaml

# Apply ClusterIssuer
kubectl apply -f ingress.yaml

# Certificates will be auto-generated
```

**Option B: Bring Your Own Certificate**

```bash
kubectl create secret tls cctv-tls-cert \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  --namespace=cctv-system
```

## 📦 Deployment Steps

### 1. Network Policies (Zero Trust)

```bash
kubectl apply -f network-policy.yaml
```

This creates:
- Default deny-all policy
- Specific allow rules for each component
- DNS resolution allowed for all pods

### 2. RBAC Configuration

```bash
kubectl apply -f rbac.yaml
```

Creates:
- Service accounts per component
- Least-privilege roles
- Pod Security Policies (if using K8s <1.25)

### 3. Database (PostgreSQL)

```bash
# Apply StatefulSet and services
kubectl apply -f postgres-statefulset.yaml

# Wait for ready
kubectl wait --for=condition=ready pod -l app=postgres --namespace=cctv-system --timeout=300s

# Verify
kubectl get statefulset postgres --namespace=cctv-system
kubectl get pvc --namespace=cctv-system
```

### 4. Cache (Redis)

```bash
kubectl apply -f redis-deployment.yaml

# Wait for ready
kubectl wait --for=condition=ready pod -l app=redis --namespace=cctv-system --timeout=120s
```

### 5. API Backend

```bash
# Build and push Docker image first
docker build -f Dockerfile.prod -t your-registry/cctv-api:v1.0.0 .
docker push your-registry/cctv-api:v1.0.0

# Update image in api-deployment.yaml
sed -i 's|cctv-api:latest|your-registry/cctv-api:v1.0.0|g' api-deployment.yaml

# Deploy
kubectl apply -f api-deployment.yaml

# Wait for ready
kubectl wait --for=condition=ready pod -l app=cctv-api --namespace=cctv-system --timeout=300s

# Check rollout
kubectl rollout status deployment/cctv-api --namespace=cctv-system
```

### 6. Ingress (External Access)

```bash
kubectl apply -f ingress.yaml

# Get external IP
kubectl get ingress --namespace=cctv-system

# Wait for certificate (can take 2-3 minutes)
kubectl get certificate --namespace=cctv-system
```

## 🔍 Verification

```bash
# Check all pods
kubectl get pods --namespace=cctv-system

# Check services
kubectl get svc --namespace=cctv-system

# Check ingress
kubectl get ingress --namespace=cctv-system

# View logs
kubectl logs -l app=cctv-api --namespace=cctv-system --tail=100

# Test health endpoint
curl https://api.your-domain.com/health/live
```

## 📊 Monitoring

### Prometheus & Grafana

```bash
# Install Prometheus Stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Default: admin / prom-operator
```

### Application Metrics

- API metrics: `http://cctv-api:8000/metrics`
- PostgreSQL metrics: Exposed on port 9187
- Redis metrics: Exposed on port 9121

## 📈 Scaling

### Horizontal Scaling (HPA)

Horizontal Pod Autoscaler is already configured:

```bash
# View HPA status
kubectl get hpa --namespace=cctv-system

# Manually scale
kubectl scale deployment cctv-api --replicas=5 --namespace=cctv-system
```

### Vertical Scaling

Update resource requests/limits in `api-deployment.yaml`

## 🔄 Updates & Rollbacks

```bash
# Update image
kubectl set image deployment/cctv-api \
  api=your-registry/cctv-api:v1.1.0 \
  --namespace=cctv-system

# Watch rollout
kubectl rollout status deployment/cctv-api --namespace=cctv-system

# Rollback if issues
kubectl rollout undo deployment/cctv-api --namespace=cctv-system

# View history
kubectl rollout history deployment/cctv-api --namespace=cctv-system
```

## 🛠️ Troubleshooting

### Pods Not Starting

```bash
# Describe pod
kubectl describe pod POD_NAME --namespace=cctv-system

# Check events
kubectl get events --namespace=cctv-system --sort-by='.lastTimestamp'

# View logs
kubectl logs POD_NAME --namespace=cctv-system --previous
```

### Database Issues

```bash
# Connect to database
kubectl exec -it postgres-0 --namespace=cctv-system -- psql -U cctv_user -d cctv_db

# Check database logs
kubectl logs postgres-0 --namespace=cctv-system
```

### Network Issues

```bash
# Test connectivity
kubectl run -it --rm debug --image=busybox --restart=Never --namespace=cctv-system -- sh
# Inside pod: wget -O- http://cctv-api:80/health

# Check network policies
kubectl get networkpolicies --namespace=cctv-system
```

## ✅ Production Checklist

- [ ] All secrets stored in external secret manager
- [ ] TLS certificates configured (Let's Encrypt or custom)
- [ ] Network policies applied
- [ ] Resource limits configured
- [ ] HPA enabled
- [ ] PodDisruptionBudget configured
- [ ] Monitoring and alerting set up
- [ ] Backup strategy configured
- [ ] Disaster recovery plan documented
- [ ] Security scanning in CI/CD
- [ ] Load testing completed
- [ ] Incident response procedures documented

## 🔒 Security Best Practices

1. **Never commit secrets to Git**
2. **Use RBAC with least privilege**
3. **Enable network policies**
4. **Run containers as non-root**
5. **Use read-only root filesystems**
6. **Regular security updates**
7. **Enable audit logging**
8. **Implement rate limiting**
9. **Use WAF (ModSecurity)**
10. **Regular penetration testing**

## 📚 Additional Resources

- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Network Policies Guide](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [RBAC Documentation](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)

## 🆘 Support

For issues or questions:
- Create an issue on GitHub
- Contact: security@company.com
- Documentation: https://docs.company.com
