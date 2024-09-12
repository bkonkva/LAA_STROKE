USE ORD_Kicska_202402007D
-- select count(*) FROM  DFLT.IcdEvents --3803819
 --select count(  distinct patientSID, DiagDateTime, Sta3n, icd10sid, diagnosis_category, ICD_Code, ICD_Classification, Origin)  from DFLT.IcdEvents -- --3803819
--#########################
	--Gathering Patient ICD EVENTS
--#########################
drop table if exists DFLT.IcdEvents 



  SELECT PatientSID, AdmitDateTime as DiagDateTime, Sta3n, d.*,'InpatientDischargeDiagnosis' Origin  
	INTO
	
	DFLT.IcdEvents 
			from  --select top 10 * from
				Src.Inpat_InpatDischargeDiagnosis a
					inner join
				[Dflt].[ICD10] d
			ON
				a.ICD10SID=d.ICD10SID 
				where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION 
	SELECT PatientSID, AdmitDateTime, Sta3n,  d.*,'InpatientFeeDiagnosis'
					  --select top 10 * 
			 FROM  Src.Inpat_InpatientFeeDiagnosis a
					inner join
						[Dflt].[ICD10] d
			ON
				a.ICD10SID=d.ICD10SID 
				where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION 
	SELECT PatientSID, PatientTransferDateTime, Sta3n, d.*,'PatientTransferDiagnosis'  	
			
					--select top 10 * 
					  FROM  Src.Inpat_PatientTransferDiagnosis a
								inner join
							[Dflt].[ICD10] d
							ON
								a.ICD10SID=d.ICD10SID 
							where 
								PatientSID in ( select PatientSID from Dflt.RH_Results3 )
			 
	UNION 
	SELECT PatientSID, SpecialtyTransferDateTime, Sta3n, d.*,'SpecialtyTransferDiagnosis'  

				--select top 10 * 
					  FROM  Src.Inpat_SpecialtyTransferDiagnosis a
								inner join
							[Dflt].[ICD10] d
								ON
									a.ICD10SID=d.ICD10SID 
									where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION 
	SELECT PatientSID, DischargeDateTime, Sta3n, d.*,'InpatientDiagnosis'   
	
					--select top 10 * 
					  FROM  Src.Inpat_InpatientDiagnosis a
					  					inner join
							[Dflt].[ICD10] d
						ON
							a.ICD10SID=d.ICD10SID 
							where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
				

	UNION
	SELECT PatientSID , VisitDateTime, Sta3n,  d.*,'VProcedureDiagnosis'  				  
					  FROM  Src.Outpat_VProcedureDiagnosis a
								inner join
							[Dflt].[ICD10] d
								ON
									a.ICD10SID=d.ICD10SID 
									where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION
	SELECT PatientSID, VisitDateTime, Sta3n,  d.*,'ICD1WorkloadVDiagnosis0SID'   				  
					  FROM  Src.Outpat_WorkloadVDiagnosis a
								inner join
							[Dflt].[ICD10] d
						ON
							a.ICD10SID=d.ICD10SID 
							where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	 
	UNION
	SELECT PatientSID, LastModifiedDateTime, Sta3n,  d.*,'ProblemList'   	
							 
						FROM  Src.Outpat_ProblemList a
							inner join
									[Dflt].[ICD10] d
								ON
									a.ICD10SID=d.ICD10SID 
							where 
							PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION
	SELECT PatientSID, VisitDateTime, Sta3n, d.*,'VDiagnosis'   				  
					  FROM  Src.Outpat_VDiagnosis_Recent a
								inner join
							[Dflt].[ICD10] d
						ON
							a.ICD10SID=d.ICD10SID 
							where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION
	SELECT PatientSID, VisitDateTime, Sta3n, d.*,'VDiagnosis'   				  
					  FROM  Src.Outpat_VDiagnosis a
								inner join
							[Dflt].[ICD10] d
						ON
							a.ICD10SID=d.ICD10SID 
							where PatientSID in ( select PatientSID from Dflt.RH_Results3 )

	UNION
	SELECT PatientSID, AdmitDateTime, Sta3n, d.*,'InpatientDischargeDiagnosis' Origin  

			from  --select top 10 * from
				Src.Inpat_InpatDischargeDiagnosis a
					inner join --SELECT * from 
				[Dflt].[ICD9] d
			ON
				a.ICD9SID=d.ICD9SID 
				where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION 
	SELECT PatientSID, AdmitDateTime, Sta3n,  d.*,'InpatientFeeDiagnosis'
					  --select top 10 * 
			 FROM  Src.Inpat_InpatientFeeDiagnosis a
					inner join
						[Dflt].[ICD9] d
			ON
				a.ICD9SID=d.ICD9SID 
				where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION 
	SELECT PatientSID, PatientTransferDateTime, Sta3n, d.*,'PatientTransferDiagnosis'  	
			
					--select top 10 * 
					  FROM  Src.Inpat_PatientTransferDiagnosis a
								inner join
							[Dflt].[ICD9] d
							ON
								a.ICD9SID=d.ICD9SID 
							where 
								PatientSID in ( select PatientSID from Dflt.RH_Results3 )
			 
	UNION 
	SELECT PatientSID, SpecialtyTransferDateTime, Sta3n, d.*,'SpecialtyTransferDiagnosis'  

				--select top 10 * 
					  FROM  Src.Inpat_SpecialtyTransferDiagnosis a
								inner join
							[Dflt].[ICD9] d
								ON
									a.ICD9SID=d.ICD9SID 
									where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION 
	SELECT PatientSID, DischargeDateTime, Sta3n, d.*,'InpatientDiagnosis'   
	
					--select top 10 * 
					  FROM  Src.Inpat_InpatientDiagnosis a
					  					inner join
							[Dflt].[ICD9] d
						ON
							a.ICD9SID=d.ICD9SID 
							where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
				

	UNION
	SELECT PatientSID , VisitDateTime, Sta3n,  d.*,'VProcedureDiagnosis'  				  
					  FROM  Src.Outpat_VProcedureDiagnosis a
								inner join
							[Dflt].[ICD9] d
								ON
									a.ICD9SID=d.ICD9SID 
									where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION
	SELECT PatientSID, VisitDateTime, Sta3n,  d.*,'ICD1WorkloadVDiagnosis0SID'   				  
					  FROM  Src.Outpat_WorkloadVDiagnosis a
								inner join
							[Dflt].[ICD9] d
						ON
							a.ICD9SID=d.ICD9SID 
							where PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	 
	UNION
	SELECT PatientSID, LastModifiedDateTime, Sta3n,  d.*,'ProblemList'   	
							 
						FROM  Src.Outpat_ProblemList a
							inner join
									[Dflt].[ICD9] d
								ON
									a.ICD9SID=d.ICD9SID 
							where 
							PatientSID in ( select PatientSID from Dflt.RH_Results3 )
	UNION
	SELECT PatientSID, VisitDateTime, Sta3n, d.*,'VDiagnosis'   --select * 				  
					  FROM  Src.Outpat_VDiagnosis_Recent a
								inner join
							[Dflt].[ICD9] d
						ON
							a.ICD9SID=d.ICD9SID 
							where PatientSID in ( select PatientSID from Dflt.RH_Results3 )

		SELECT PatientSID, VisitDateTime, Sta3n, d.*,'VDiagnosis'   --select * 				  
					  FROM  Src.Outpat_VDiagnosis a
								inner join
							[Dflt].[ICD9] d
						ON
							a.ICD9SID=d.ICD9SID 
							where PatientSID in ( select PatientSID from Dflt.RH_Results3 )


--######################################################################################### use vaci_synpat

DROP TABLE IF EXISTS #ICD;
DROP TABLE IF EXISTS #Final;
DROP TABLE IF EXISTS #Stroke;
DROP TABLE IF EXISTS #StrokeCnt;


select PatientSID, cast(convert(varchar,DiagDateTime,1) as Date) as StrkDate 
,count(*) over ( partition by patientsid ) TotalStrokes
,count(*) over ( partition by patientsid  order by cast(convert(varchar,DiagDateTime,1) as Date) ) TotalStrkCnt
, ROW_NUMBER( ) over( partition by patientsid order by patientsid) rownum
INTO #StrokeCnt
from dflt.IcdEvents 
where Diagnosis_Category = 'Stroke'
group by PatientSID, cast(convert(varchar,DiagDateTime,1) as Date)
order by 1,3  desc offset 0 rows

--	select * From #StrokeCnt order by 1
--################################################## use cdwwork 
;with cte as
(
select *

,ROW_NUMBER() over ( partition by e.patientsid, e.Diagnosis_Category  order by patientsid,e.DiagDateTime ) rownum
,count(*) over ( partition by e.patientsid, e.Diagnosis_Category  order by patientsid,e.DiagDateTime ) DiagCnt
from DFLT.IcdEvents e
order by PatientSID offset 0 rows
)
,diagcte as
(
select * from cte where rownum = 1
)
select * into #ICD from diagcte



;with cte as (
		select R.PatientSID, R.RadiologyExamSID ,R.ExamDateTime 
			


		from Dflt.RH_Results3 R

		left outer join --select * from
             #ICD I
				on R.PatientSID = I.Patientsid
		where --r.RadiologyExamSID = 800004175320 and
		 iif(R.ExamDateTime >= ISNULL(I.DiagDateTime,'01/01/01'),'AFTER','BEFORE') ='AFTER'

UNION

select R.PatientSID, R.RadiologyExamSID,R.ExamDateTime 
		
		

			,IIF(I.Diagnosis_Category = 'Stroke','StrokeOutcomeFlg','TIAOutcomeFlg'),i.rownum icd_rownum
			

		from Dflt.RH_Results3 R

		left outer join --select * from
             #ICD I
				on R.PatientSID = I.Patientsid
		where-- r.RadiologyExamSID = 800004175320 and
		 iif(R.ExamDateTime >= ISNULL(I.DiagDateTime,'01/01/01'),'AFTER','BEFORE') ='BEFORE'
		and Diagnosis_Category in ( 'Stroke', 'TIA')
		
UNION

select R.PatientSID, R.RadiologyExamSID,R.ExamDateTime 
			,'NoOutcome',i.rownum icd_rownum
		
	

		from Dflt.RH_Results3 R --23,733,10,652

		left outer join --select * from
             #ICD I
				on R.PatientSID = I.Patientsid
		where-- r.RadiologyExamSID = 800004175320 and
		 iif(R.ExamDateTime >= ISNULL(I.DiagDateTime,'01/01/01'),'AFTER','BEFORE') ='BEFORE'
		and Diagnosis_Category not in ( 'Stroke', 'TIA')

				)
				
				
--#########################
	--Summarizing Patientg Events
--#########################



select  * into #Final from cte 


		PIVOT(
       COUNT(icd_rownum) 
	  --  COUNT(riskflg)
        FOR Diagnosis_Category 
                IN (
					[TIA]
					,[Atrial Fibrillation or Atrial Flutter]
					,[Stroke]
					,[Vascular Disease]
					,[Hypertension]
					,[CHF and other forms]
					,[Diabetes]
					,[Personal history of transient ischmeic attach and cerebral infarction] 
					,[StrokeOutcomeFlg]
					,[TIAOutcomeFlg]
					,[NoOutcomeFlg]
				   )
			) AS pivot_table;
	
		ALTER TABLE  #Final add Stroke_Icd_Code varchar(500)
		,TIA_Icd_Code varchar(500)
		, Atrial_Icd_Code  varchar(500)
		, Vascular_Icd_Code  varchar(500)
		, Hypertension_Icd_Code  varchar(500)
		, CHF_Icd_Code  varchar(500)
		, Diab_Icd_Code  varchar(500)
		,Ischm_Icd_Code  varchar(500)
		

		UPDATE F SET F.TIA_Icd_Code = D.ICD_Code 
			FROM #Final F
			INNER JOIN #ICD D 
					on F.PatientSID = D.PatientSID
			Where D.Diagnosis_Category = 'TIA'


			
		UPDATE F SET F.Atrial_Icd_Code = D.ICD_Code 
			FROM #Final F
			INNER JOIN #ICD D 
					on F.PatientSID = D.PatientSID
			Where D.Diagnosis_Category = 'Atrial Fibrillation or Atrial Flutter'
			
		UPDATE F SET F.Stroke_Icd_Code = D.ICD_Code 
			FROM #Final F
			INNER JOIN #ICD D 
					on F.PatientSID = D.PatientSID
			Where D.Diagnosis_Category = 'Stroke'
			
		UPDATE F SET F.Vascular_Icd_Code = D.ICD_Code 
			FROM #Final F
			INNER JOIN #ICD D 
					on F.PatientSID = D.PatientSID
			Where D.Diagnosis_Category = 'Vascular Disease'
		
		UPDATE F SET F.Hypertension_Icd_Code = D.ICD_Code 
			FROM #Final F
			INNER JOIN #ICD D 
					on F.PatientSID = D.PatientSID
			Where D.Diagnosis_Category = 'Hypertension'
			
		UPDATE F SET F.CHF_Icd_Code = D.ICD_Code 
			FROM #Final F
			INNER JOIN #ICD D 
					on F.PatientSID = D.PatientSID
			Where D.Diagnosis_Category = 'CHF and other forms'
			
		UPDATE F SET F.Diab_Icd_Code = D.ICD_Code 
			FROM #Final F
			INNER JOIN #ICD D 
					on F.PatientSID = D.PatientSID
			Where D.Diagnosis_Category = 'Diabetes'
			
		UPDATE F SET F.Ischm_Icd_Code = D.ICD_Code 
			FROM #Final F
			INNER JOIN #ICD D 
					on F.PatientSID = D.PatientSID
			Where D.Diagnosis_Category = 'Personal history of transient ischmeic attach and cerebral infarction'
			--select * into dflt.ComorditiesTableOLD  From dflt.ComorditiesTable
drop table if exists dflt.ComorditiesTable;
SELECT  p.PatientICN, p.age,p.Gender , --isnull(s.TotalStrokes,0) TotalStrokes,
		-- p.birthdatetime,f.ExamDateTime
		 datediff(YEAR,p.birthdatetime,f.examdatetime) Age_at_Exam,
		
			IIF(datediff(YEAR,p.birthdatetime,f.examdatetime)>=65,
				IIF(datediff(YEAR,p.birthdatetime,f.examdatetime) >= 75,2,1
					),
			0
			)
		+
	[TIA] +	[Atrial Fibrillation or Atrial Flutter] +2*[Stroke]+[Vascular Disease]+[Hypertension]+[CHF and other forms]+	[Diabetes]+	[Personal history of transient ischmeic attach and cerebral infarction] as Chads2VascScore
	--,[TIA] +	[Atrial Fibrillation or Atrial Flutter] +2*[Stroke]+[Vascular Disease]+[Hypertension]+[CHF and other forms]+	[Diabetes]+	[Personal history of transient ischmeic attach and cerebral infarction] as TentativeScore
	--select Count(*) FROM dflt.ComorditiesTable --244165
	,f.*
into dflt.ComorditiesTable
-- select count( distinct f.patientsid )
 from #final f
 --where patientsid = 17461950
inner join Src.SPatient_SPatient p
		on p.PatientSID = f.PatientSID
--left outer join --select * FROM
--	#StrokeCnt s order by 1,5
--		on		f.patientsid = s.patientsid
--where s.rownum =1

 order by f.PatientSID



  
UPDATE dflt.ComorditiesTable set Atrial_Icd_Code = NULL where [Atrial Fibrillation or Atrial Flutter] = 0
UPDATE dflt.ComorditiesTable set CHF_Icd_Code = NULL WHERE [CHF and other forms] = 0
UPDATE dflt.ComorditiesTable set Diab_Icd_Code = NULL  WHERE Diabetes = 0
UPDATE dflt.ComorditiesTable set Hypertension_Icd_Code = NULL  WHERE Hypertension = 0
UPDATE dflt.ComorditiesTable set Ischm_Icd_Code = NULL  WHERE  [Personal history of transient ischmeic attach and cerebral infarction] = 0
UPDATE dflt.ComorditiesTable set Vascular_Icd_Code = NULL  WHERE [Vascular Disease] = 0

 