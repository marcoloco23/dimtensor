{{/*
Expand the name of the chart.
*/}}
{{- define "dimtensor.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "dimtensor.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "dimtensor.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "dimtensor.labels" -}}
helm.sh/chart: {{ include "dimtensor.chart" . }}
{{ include "dimtensor.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "dimtensor.selectorLabels" -}}
app.kubernetes.io/name: {{ include "dimtensor.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "dimtensor.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "dimtensor.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the full image name
*/}}
{{- define "dimtensor.image" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.image.registry }}
{{- $repository := .Values.image.repository }}
{{- $tag := .Values.image.tag | default .Chart.AppVersion }}
{{- $variant := .Values.image.variant }}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository (ternary $variant (printf "%s-%s" $tag $variant) (eq $variant "latest")) }}
{{- else }}
{{- printf "%s:%s" $repository (ternary $variant (printf "%s-%s" $tag $variant) (eq $variant "latest")) }}
{{- end }}
{{- end }}

{{/*
Return true if GPU is enabled
*/}}
{{- define "dimtensor.gpu.enabled" -}}
{{- .Values.resources.gpu.enabled }}
{{- end }}

{{/*
GPU node selector
*/}}
{{- define "dimtensor.gpu.nodeSelector" -}}
{{- if .Values.resources.gpu.enabled }}
accelerator: nvidia-gpu
{{- end }}
{{- end }}

{{/*
GPU tolerations
*/}}
{{- define "dimtensor.gpu.tolerations" -}}
{{- if .Values.resources.gpu.enabled }}
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
{{- end }}
{{- end }}

{{/*
Return the proper workload kind
*/}}
{{- define "dimtensor.workloadKind" -}}
{{- if eq .Values.workloadType "job" }}
Job
{{- else if eq .Values.workloadType "cronjob" }}
CronJob
{{- else }}
Deployment
{{- end }}
{{- end }}
