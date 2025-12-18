# Kubernetes Production Deployment Guide

## 🚀 Quick Start

### Prerequisites

1. **Kubernetes Cluster** (v1.25+)
   - Managed: EKS, GKE, AKS
   - Self-hosted: kubeadm, k3s, kind (dev only)

2. **Tools Required**
   ```bash
   kubectl version --client
   helm version
   ```

3. **NGINX Ingress Controller**
   ```bash
   helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
   helm install ingress-nginx ingress-nginx/ingress-nginx \
     --namespace ingress-nginx --create-namespace
   ```

4. **cert-manager** (for TLS)
   ```bash
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
   ```

---

## 📦 Deployment Steps

### 1. Create Namespace
```bash
kubectl apply -f namespace.yaml
```

### 2. Apply RBAC Policies
```bash
kubectl apply -f rbac.yaml
```

### 3. Apply Network Policies (Zero-Trust)
```bash
kubectl apply -f network-policy.yaml
```

### 4. Create Secrets

**⚠️ NEVER commit secrets to Git!**

```bash
# Generate strong passwords
export DB_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export ENCRYPTION_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Create secrets
kubectl create secret generic cctv-db-credentials \
  --from-literal=url="postgresql://cctv_user:${DB_PASSWORD}@postgresql-service:5432/cctv_db?sslmode=require" \
  --from-literal=username="cctv_user" \
  --from-literal=password="${DB_PASSWORD}" \
  --from-literal=database="cctv_db" \
  -n cctv-detection

kubectl create secret generic cctv-redis-credentials \
  --from-literal=url="redis://:${REDIS_PASSWORD}@redis-service:6379/0" \
  --from-literal=password="${REDIS_PASSWORD}" \
  -n cctv-detection

kubectl create secret generic cctv-api-keys \
  --from-literal=secret-key="${SECRET_KEY}" \
  --from-literal=encryption-key="${ENCRYPTION_KEY}" \
  --from-literal=jwt-secret="$(openssl rand -base64 32)" \
  -n cctv-detection

kubectl create secret generic postgresql-credentials \
  --from-literal=postgres-password="${DB_PASSWORD}" \
  --from-literal=replication-password="$(openssl rand -base64 32)" \
  -n cctv-detection
```

**Production:** Use **Sealed Secrets**, **External Secrets Operator**, or **HashiCorp Vault**

### 5. Deploy PostgreSQL
```bash
kubectl apply -f postgresql-statefulset.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgresql -n cctv-detection --timeout=300s
```

### 6. Deploy Redis
```bash
kubectl apply -f redis-deployment.yaml

# Wait for Redis
kubectl wait --for=condition=ready pod -l app=redis -n cctv-detection --timeout=180s
```

### 7. Apply Resource Quotas
```bash
kubectl apply -f resource-quotas.yaml
```

### 8. Deploy API
```bash
# Build and push Docker image
docker build -f Dockerfile.production -t YOUR_REGISTRY/cctv-api:v1.0.0 .
docker push YOUR_REGISTRY/cctv-api:v1.0.0

# Update image in api-deployment.yaml
# Then apply:
kubectl apply -f api-deployment.yaml

# Wait for API
kubectl wait --for=condition=ready pod -l app=cctv-api -n cctv-detection --timeout=300s
```

### 9. Deploy Ingress
```bash
# Update domain in ingress.yaml
# Update email in ClusterIssuer
kubectl apply -f ingress.yaml

# Check certificate
kubectl get certificate -n cctv-detection
kubectl describe certificate cctv-tls-cert -n cctv-detection
```

### 10. Verify Deployment
```bash
# Check all pods
kubectl get pods -n cctv-detection

# Check services
kubectl get svc -n cctv-detection

# Check ingress
kubectl get ingress -n cctv-detection

# Test API
curl -k https://api.cctv-detection.example.com/health
```

---

## 🔒 Security Checklist

### Pre-Deployment
- [ ] Secrets stored securely (Vault/External Secrets)
- [ ] TLS certificates configured
- [ ] Network policies applied
- [ ] RBAC roles configured
- [ ] Resource quotas set
- [ ] Pod Security Standards enforced
- [ ] Image vulnerability scanning completed

### Post-Deployment
- [ ] Ingress accessible over HTTPS only
- [ ] Database connections encrypted
- [ ] API authentication working
- [ ] Rate limiting active
- [ ] Logging configured
- [ ] Monitoring/alerting setup
- [ ] Backup jobs scheduled
- [ ] Disaster recovery tested

---

## 🎯 Production Best Practices

### 1. **Image Management**
```bash
# Scan images before deployment
trivy image YOUR_REGISTRY/cctv-api:v1.0.0

# Use image digests instead of tags
image: YOUR_REGISTRY/cctv-api@sha256:abc123...
```

### 2. **Secrets Management**
```bash
# Use External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets-system --create-namespace
```

### 3. **Monitoring**
```bash
# Install Prometheus & Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace
```

### 4. **Logging**
```bash
# Install EFK/ELK stack
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch -n logging --create-namespace
helm install kibana elastic/kibana -n logging
helm install filebeat elastic/filebeat -n logging
```

### 5. **Backup Strategy**
```bash
# Install Velero for cluster backups
velero install \
  --provider aws \
  --bucket my-backup-bucket \
  --secret-file ./credentials-velero

# Schedule backups
velero schedule create daily-backup --schedule="0 2 * * *"
```

---

## 🔧 Maintenance

### Rolling Updates
```bash
# Update API image
kubectl set image deployment/cctv-api \
  api=YOUR_REGISTRY/cctv-api:v1.1.0 \
  -n cctv-detection

# Monitor rollout
kubectl rollout status deployment/cctv-api -n cctv-detection

# Rollback if needed
kubectl rollout undo deployment/cctv-api -n cctv-detection
```

### Database Maintenance
```bash
# Manual backup
kubectl exec -it postgresql-0 -n cctv-detection -- \
  pg_dump -U cctv_user cctv_db > backup.sql

# Restore
kubectl exec -i postgresql-0 -n cctv-detection -- \
  psql -U cctv_user cctv_db < backup.sql
```

### Scaling
```bash
# Manual scaling
kubectl scale deployment cctv-api --replicas=5 -n cctv-detection

# HPA is already configured in api-deployment.yaml
# Check autoscaling
kubectl get hpa -n cctv-detection
```

---

## 🚨 Troubleshooting

### Pod Not Starting
```bash
# Check pod status
kubectl describe pod POD_NAME -n cctv-detection

# Check logs
kubectl logs POD_NAME -n cctv-detection

# Check previous logs (if pod restarted)
kubectl logs POD_NAME -n cctv-detection --previous
```

### Network Issues
```bash
# Test DNS
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup postgresql-service.cctv-detection.svc.cluster.local

# Test connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- nc -zv postgresql-service 5432
```

### Permission Issues
```bash
# Check RBAC
kubectl auth can-i get pods --as=system:serviceaccount:cctv-detection:cctv-api-sa -n cctv-detection

# Check security context
kubectl exec POD_NAME -n cctv-detection -- id
```

---

## 📊 Monitoring Queries

### Prometheus Queries
```promql
# API Request Rate
rate(http_requests_total[5m])

# API Error Rate
rate(http_requests_total{status=~"5.."}[5m])

# Pod CPU Usage
sum(rate(container_cpu_usage_seconds_total{namespace="cctv-detection"}[5m])) by (pod)

# Pod Memory Usage
sum(container_memory_usage_bytes{namespace="cctv-detection"}) by (pod)
```

---

## 🌐 External Access

### LoadBalancer (Cloud)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: cctv-api-external
  namespace: cctv-detection
spec:
  type: LoadBalancer
  selector:
    app: cctv-api
  ports:
  - port: 443
    targetPort: 8000
```

### NodePort (Development Only)
```bash
# Not recommended for production
kubectl expose deployment cctv-api --type=NodePort --port=8000 -n cctv-detection
```

---

## 📚 Additional Resources

- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [cert-manager Documentation](https://cert-manager.io/docs/)

---

## 🆘 Support

For issues or questions:
1. Check logs: `kubectl logs -n cctv-detection`
2. Review events: `kubectl get events -n cctv-detection --sort-by='.lastTimestamp'`
3. Open GitHub issue with logs and deployment details
