#!/bin/bash
# Generate Docker secrets for docker-compose

set -e

echo "=== Generating Docker Secrets ==="
echo

# Create secrets directory
mkdir -p secrets

# Generate database password
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
echo "$DB_PASSWORD" > secrets/db_password.txt
echo "✅ Generated database password: secrets/db_password.txt"

# Generate Grafana password
GRAFANA_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
echo "$GRAFANA_PASSWORD" > secrets/grafana_password.txt
echo "✅ Generated Grafana password: secrets/grafana_password.txt"

# Set proper permissions (read-only for owner)
chmod 400 secrets/*.txt

echo
echo "📝 Generated Credentials:"
echo "   Database Password: $DB_PASSWORD"
echo "   Grafana Password: $GRAFANA_PASSWORD"
echo
echo "⚠️  Save these credentials securely!"
echo "🔒 Secret files have been created with read-only permissions"
echo
echo "Next steps:"
echo "   1. Update .env.production with DATABASE_URL using this password"
echo "   2. Run: docker-compose up -d"
echo "   3. Access Grafana at http://localhost:3000 (admin / <grafana_password>)"
