	USE ORD_Kicska_202402007D
	
	-- Add Sta3n's inclusion criteria, leave null if your study is scope is all stations
	DECLARE @Sta3ns as VARCHAR(MAX) = '663,463,648,531,687,668,653,692,399';
	-- For this example our remaining inclustion criteria are the procedure CPT code 
	-- and a start & end date. This script can be adapted to your specific inclusion criteria.
	DECLARE @CPTCodes as VARCHAR(MAX) = '71270,71275';
	DECLARE @StartDateTime as datetime2(0) = '2011-01-01';
	DECLARE @EndDateTime as datetime2(0) = '2023-09-30';

	DECLARE @Sta3ns_parsed as TABLE(Sta3n SMALLINT PRIMARY KEY NOT NULL);

	IF	@Sta3ns IS NULL
		BEGIN
			INSERT INTO @Sta3ns_parsed SELECT Sta3n FROM CDWWork.Dim.Sta3n;
		END
	ELSE
		BEGIN
			INSERT INTO @Sta3ns_parsed SELECT DISTINCT CAST(TRIM(value) as SMALLINT) AS Sta3n FROM STRING_SPLIT(@Sta3ns, ',');
		END;

	DROP TABLE IF EXISTS #RadiologyExam;
	WITH RadiologyProcedure AS (
		SELECT
			p.RadiologyProcedureSID
			,p.Sta3n
			,p.RadiologyProcedure
			,p.ContrastMediaUsedFlag
			,c.CPTCode
			,c.CPTDescription		
		FROM 
			CDWWork.Dim.RadiologyProcedure AS p
		LEFT JOIN 
			CDWWork.Dim.CPT AS c
		ON 
			c.CPTSID = p.CPTSID
		WHERE
			c.CPTCode IN(SELECT DISTINCT CAST(TRIM(value) as CHAR(5)) FROM STRING_SPLIT(@CPTCodes, ','))
			AND p.Sta3n IN(SELECT Sta3n FROM @Sta3ns_parsed)
	)
	SELECT
		e.RadiologyExamSID
		,e.RadiologyExamIEN
		,e.Sta3n
		,e.PatientSID
		,e.ExamDateTime
		,e.RadiologyProcedureSID
		,p.RadiologyProcedure
		,e.ContrastMediaUsedFlag
		,p.CPTCode
		,p.CPTDescription		
		,e.CaseNumber
		,ISNULL(e.SiteAccessionNumber,CONCAT(e.Sta3n,'-', replace(convert(varchar, isnull(ExamDateTime,RequestedDateTime), 10),'-',''),'-', CAST(CaseNumber AS INT))) SiteAccessionNumber
		,e.StudyInstanceUID	
		,e.RadiologyNuclearMedicineReportSID
	INTO
		#RadiologyExam
	FROM
		Src.Rad_RadiologyExam AS e
	INNER JOIN
		RadiologyProcedure AS p
	ON
		e.RadiologyProcedureSID = p.RadiologyProcedureSID
	LEFT JOIN
		CDWWork.Dim.RadiologyExamStatus AS st
	ON
		st.RadiologyExamStatusSID = e.RadiologyExamStatusSID
	WHERE
		e.ExamDateTime BETWEEN @StartDateTime AND @EndDateTime
		AND st.RadiologyExamStatus = 'COMPLETE';

	--## Saving Imaging Information For Downstream Analytics ##

		drop table if exists [Dflt].[RH_Results3]

		select * into [Dflt].[RH_Results3] FROM #RadiologyExam